from functools import partial
import os
from typing import Callable

import numpy as np

from flax import linen as nn, optim
import jax
from jax import lax
import jax.numpy as jnp

from flax.core.frozen_dict import FrozenDict

from alpa.model.model_util import (FlaxBaseModelOutput,
                                   FlaxBaseModelOutputWithPooling,
                                   FlaxBertForPreTrainingOutput,
                                   FlaxMaskedLMOutput,
                                   FlaxSequenceClassifierOutput, TrainState)
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary


from bert_model import BertConfig, FlaxBertForSequenceClassificationModule

checkpoint_path = "/home/shangyin/Research/SparTA/script/checkpoints/bert"
import torch

def convert_to_sparsity(t):
    return [t.shape, float(1 - jnp.count_nonzero(t)/jnp.size(t))]

def explore_params(params, prefix=""):
    res = []
    for k, v in params.items():
        if isinstance(v, FrozenDict):
            s = f"{k}.{explore_params(v, prefix+'.'+k if len(prefix) else k)}"
        else:
            print(prefix + "." +  k + " " + str(convert_to_sparsity(v)))


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
    explore_params(params)

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
