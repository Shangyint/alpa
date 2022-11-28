from functools import partial
import os
from typing import Callable

import numpy as np

from flax import linen as nn, optim
import jax
from jax import lax
import jax.numpy as jnp
from jax.tools.jax_to_ir import jax_to_ir



from flax.core.frozen_dict import FrozenDict

from alpa.model.model_util import (FlaxBaseModelOutput,
                                   FlaxBaseModelOutputWithPooling,
                                   FlaxBertForPreTrainingOutput,
                                   FlaxMaskedLMOutput,
                                   FlaxSequenceClassifierOutput, TrainState)
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary


from bert_model import BertConfig, FlaxBertForSequenceClassificationModule

from transformers import BertTokenizer
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from tqdm import tqdm, trange

import torch

checkpoint_path = "/home/shangyin/Research/SparTA/script/checkpoints/bert"
data_dir = '/home/shangyin/Research/SparTA/script/checkpoints/bert/glue_data/QQP'
model_name_or_path = '../training/result/qqp_partial/coarse_0.3/checkpoint-220000/'
max_seq_length= 128


def convert_to_sparsity(t):
    return [t.shape, float(1 - jnp.count_nonzero(t)/jnp.size(t))]

def explore_params(params, prefix=""):
    res = []
    for k, v in params.items():
        if isinstance(v, FrozenDict):
            s = f"{k}.{explore_params(v, prefix+'.'+k if len(prefix) else k)}"
        else:
            print(prefix + "." +  k + " " + str(convert_to_sparsity(v)))



def load_and_cache_examples(task, tokenizer, evaluate=False):
    
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if task in ["mnli",
                    "mnli-mm"] and args.model_type in ["roberta",
                                                       "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(
                data_dir) if evaluate else processor.get_train_examples(
                data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.int32)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.int32)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.int32)
    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.int32)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_token_type_ids,
        all_labels)
    return dataset

def evaluate(model, tokenizer, params, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ['qqp']
    eval_outputs_dirs = './tmp'
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            drop_last=True,
            batch_size=32)

        # Eval!
        # print(f"***** Running evaluation {prefix} *****")
        # print(f"  Num examples = {len(eval_dataset)}")
        # print(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        loss = 0.0

        @partial(jax.jit, static_argnums=(1,))
        def eval_step(batch, apply_fn):
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]}
            inputs["position_ids"] = jnp.ones((32, max_seq_length), dtype=jnp.int32)
            
            inputs["token_type_ids"] = batch[2]
                    # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = apply_fn(params, **inputs)
            logits = outputs[0]
            labels_data = batch[3]
            label_mask = jnp.where(labels_data > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(labels_data, logits.shape[-1])

            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()

        flag = True
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if flag:
                flag = False
                s = jax_to_ir(eval_step, [("batch", batch)], constants={"apply_fn": model.apply}, format='HLO')[1]
                with open("jax_ir", "w") as f:
                    f.write(s)
            batch = tuple(t.numpy() for t in batch)
            eval_step(batch, model.apply)

    return results

def convert_from_pytorch(pt_state, config: BertConfig):
    jax_state = dict()

    # Need to change some parameters name to match Flax names so that we don't have to fork any layer
    for key, tensor in pt_state.items():
        key = "params." + key
        # Key parts
        key_parts = set(key.split("."))
        tensor = tensor.numpy()
        if 1 >= convert_to_sparsity(tensor)[1] >= 0.8:
            # useful tensor to copy
            if "dense.module.weight" in key:
                key = key.replace("module.weight", "kernel")
                jax_state[key] = tensor


            # SelfAttention needs also to replace "weight" by "kernel"
            if {"query", "key", "value"} & key_parts:
                # Flax SelfAttention decomposes the heads (num_head, size // num_heads)
                if "bias" in key:
                    jax_state[key] = tensor
                elif "weight":
                    key = key.replace("weight", "kernel")
                    tensor = tensor
                    jax_state[key] = tensor

            # There are some transposed parameters w.r.t their PyTorch counterpart
            if "intermediate.dense.kernel" in key or "output.dense.kernel" in key or "transform.dense.kernel" in key:
                jax_state[key] = tensor.T

            # Self Attention output projection needs to be transposed
            if "out.kernel" in key:
                jax_state[key] = tensor

            # Pooler needs to transpose its kernel
            if "pooler.dense.kernel" in key:
                jax_state[key] = tensor.T


            # Handle LayerNorm conversion
            if "LayerNorm" in key:

                # Replace LayerNorm by layer_norm
                
                new_key = key
                if "weight" in key:
                    new_key = new_key.replace("weight", "scale")
                # elif "bias" in key:
                #     new_key = new_key.replace("bias", "beta")

                jax_state[new_key] = tensor
    
    # for k, v in jax_state.items():
    #     if isinstance(v, torch.Tensor):
    #         print(k)
    #         jax_state[k] = v.numpy()
    for k, v in jax_state.items():
        print(k, v.shape)
    return jax_state


def flatten(p, label=None):
  if isinstance(p, FrozenDict):
    for k, v in p.items():
      yield from flatten(v, k if label is None else f"{label}.{k}")
  else:
    yield (label, p)

def load_from_pytorch_state(flattened_params, jax_state):
    for k, v in jax_state.items():
        if k in flattened_params:
            print(f"copying {k} with {v.shape} to {flattened_params[k].shape}")
            flattened_params[k] = jax_state[k]
        else:
            print(f"not found {k}")
    return flattened_params

def jax_state_qkv_combined(jax_state):
    return_state = dict(jax_state)
    qvk_kernel = dict()
    qvk_bias = dict()
    for k, v in jax_state.items():
        k_split = k.split(".")
        key_parts = set(k_split)
        if tmp := ({"query", "key", "value"} & key_parts):
            if "kernel" in k:
                del return_state[k]
                qvk_kernel.setdefault(k_split[k_split.index("layer")+1], {})[tmp.pop()] = v
            elif "bias" in k:
                del return_state[k]
                qvk_bias.setdefault(k_split[k_split.index("layer")+1], {})[tmp.pop()] = v
    
    for k, v in qvk_kernel.items():
        return_state[f"params.bert.encoder.layer.{k}.attention.self.qvk_combined.kernel"] = jnp.concatenate([v["query"], v["value"], v["key"]], axis=1)
    
    for k, v in qvk_bias.items():
        return_state[f"params.bert.encoder.layer.{k}.attention.self.qvk_combined.bias"] = jnp.concatenate([v["query"], v["value"], v["key"]], axis=0)

    return return_state   


from collections import defaultdict
from functools import reduce
from operator import getitem 

def getitem_nested(d, keys):
    return reduce(getitem, keys, d)

def default_to_frozen(d):
    if isinstance(d, defaultdict):
        return FrozenDict({k: default_to_frozen(v) for k, v in d.items()})
    else:
        return d

def convert_flattend_params(flattened_params):
    tree = lambda: defaultdict(tree)
    return_state = tree()
    for k, v in flattened_params.items():
        * keys, final_key = k.split('.')
        getitem_nested(return_state, keys)[final_key] = v
    return default_to_frozen(return_state)

def test_bert_sparse():
    config = BertConfig(num_labels=2)
    batch_size = 32
    seq_len = 128
    # Init model and optimizer w dummy inputs
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    pystate = torch.load(os.path.join(checkpoint_path, "bert_coarse_no_propagation_pytorch.pth"))
    jax_state = convert_from_pytorch(pystate, config)
    # explore_params(jax_state)


    model = FlaxBertForSequenceClassificationModule(config)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, input_ids, attention_mask, token_type_ids,
                        position_ids)
    flattened_params = dict(flatten(params))
    jax_state = jax_state_qkv_combined(jax_state)
    sparsified_params = load_from_pytorch_state(flattened_params, jax_state)

    params = convert_flattend_params(sparsified_params)
    # explore_params(params)

    token = BertTokenizer.from_pretrained('/home/shangyin/Research/SparTA/script/checkpoints/bert/checkpoints/finegrained/checkpoint-220000')
    evaluate(model, token, params)

if __name__ == "__main__":
    test_bert_sparse()


# def convert_from_pytorch(pt_state, config: BertConfig):
#     jax_state = dict(pt_state)

#     # Need to change some parameters name to match Flax names so that we don't have to fork any layer
#     for key, tensor in pt_state.items():
#         # Key parts
#         key_parts = set(key.split("."))

#         # Every dense layer has "kernel" parameters instead of "weight"
#         if "dense.weight" in key:
#             del jax_state[key]
#             key = key.replace("weight", "kernel")
#             jax_state[key] = tensor.numpy()

#         if "decoder.weight" in key:
#             del jax_state[key]
#             key = key.replace("weight", "kernel")
#             jax_state[key] = tensor.T.numpy()

#         # SelfAttention needs also to replace "weight" by "kernel"
#         if {"query", "key", "value"} & key_parts:

#             # Flax SelfAttention decomposes the heads (num_head, size // num_heads)
#             if "bias" in key:
#                 jax_state[key] = tensor.reshape((config.num_attention_heads, -1))
#             elif "weight":
#                 del jax_state[key]
#                 key = key.replace("weight", "kernel")
#                 tensor = tensor
#                 jax_state[key] = tensor.numpy()

#         # SelfAttention output is not a separate layer, remove one nesting
#         if "attention.output.dense" in key:
#             del jax_state[key]
#             key = key.replace("attention.output.dense", "attention.self.out")
#             jax_state[key] = tensor.numpy()

#         # SelfAttention output is not a separate layer, remove nesting on layer norm
#         if "attention.output.LayerNorm" in key:
#             del jax_state[key]
#             key = key.replace("attention.output.LayerNorm", "attention.LayerNorm")
#             jax_state[key] = tensor.numpy()

#         # There are some transposed parameters w.r.t their PyTorch counterpart
#         if "intermediate.dense.kernel" in key or "output.dense.kernel" in key or "transform.dense.kernel" in key:
#             jax_state[key] = tensor.T.numpy()

#         # Self Attention output projection needs to be transposed
#         if "out.kernel" in key:
#             jax_state[key] = tensor.numpy()

#         # Pooler needs to transpose its kernel
#         if "pooler.dense.kernel" in key:
#             jax_state[key] = tensor.T.numpy()

#         # Hack to correctly load some pytorch models
#         # if "predictions.bias" in key:
#         #     del jax_state[key]
#         #     jax_state[".".join(key.split(".")[:2]) + ".decoder.bias"] = tensor.numpy()

#         # Handle LayerNorm conversion
#         if "LayerNorm" in key:
#             del jax_state[key]

#             # Replace LayerNorm by layer_norm
            
#             new_key = key
#             if "weight" in key:
#                 new_key = new_key.replace("weight", "scale")
#             elif "bias" in key:
#                 new_key = new_key.replace("bias", "beta")

#             jax_state[new_key] = tensor.numpy()
    
#     for k, v in jax_state.items():
#         if isinstance(v, torch.Tensor):
#             print(k)
#             jax_state[k] = v.numpy()

#     return jax_state
