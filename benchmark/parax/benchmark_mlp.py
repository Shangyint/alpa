"""Benchmark MLP."""
import argparse
import os
import pickle
import timeit

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim
import ray

from parax import parallelize, set_parallelize_options, testing, global_config, DeviceCluster
from parax.testing import assert_only_has_allreduce
from parax.util import write_tsv

MB = 1024 ** 2
GB = 1024 ** 3


def compute_data_parallel_cost(optimizer, logical_mesh, physical_mesh):
    """For debugging usage."""
    cost = 0
    for size in [9216, 2304] * 4 + [2304 * 9216] * 8:
        cost += physical_mesh.prof_result.estimate_all_reduce(
            ((0,1,2,3),), size, "float32")
    print("Data-parallel", cost)

    cost = 0
    for size in [8192*2304] * 7 + [4608, 2304] * 4 + \
            [2304*4608] * 8:
        cost += physical_mesh.prof_result.estimate_all_reduce(
            ((0,1),(2,3),), size, "float32")
    print("Hybrid-parallel", cost)
    exit(0)


def benchmark_mlp_one_case(benchmark_case, use_profiling):
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, dp_size, tensor_mp_size =\
        benchmark_case

    class Model(nn.Module):
        hidden_size: int
        num_layers: int

        @nn.compact
        def __call__(self, x):
            for i in range(self.num_layers):
                x = nn.Dense(features=self.hidden_size * 4)(x)
                #x = nn.gelu(x)
                x = nn.Dense(features=self.hidden_size)(x)
            return x

    # Mesh configs
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh()
    assert physical_mesh.total_devices == dp_size * tensor_mp_size
    logical_mesh = physical_mesh.get_logical_mesh([dp_size, tensor_mp_size])
    set_parallelize_options(devices=logical_mesh)
                            #search_logical_mesh_shape=True,
                            #mesh_shape_search_mode="measurement")

    if use_profiling:
        filename = physical_mesh.get_signature() + ".prof.pkl"
        if os.path.exists(filename):
            print(f"Load saved profiling results from {filename}")
            physical_mesh.load_profiling_result(filename)
            physical_mesh.prof_result.multiply_scale(1e7)
        else:
            physical_mesh.profile_collective("all-reduce")
            print(f"Save profiling results to {filename}")
            physical_mesh.save_profiling_result(filename)

    @parallelize
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Prepare model and input
    batch = {
        "x": jnp.ones((batch_size, seq_len, hidden_size)),
        "y": jnp.ones((batch_size, seq_len, hidden_size)),
    }
    model = Model(hidden_size=hidden_size, num_layers=num_layers)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, batch["x"])
    optimizer = optim.GradientDescent(1e-2).create(params)
    optimizer, batch = train_step.preshard_dynamic_args(optimizer, batch, model.apply)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]

        optimizer = train_step(optimizer, batch, model.apply)
        physical_mesh.sync_workers()

        closure[0] = optimizer

    # Benchmark time cost
    func()
    stmt = "func()"
    repeat = 2
    number = args.number
    costs = np.array(timeit.repeat(stmt, globals={**globals(), **locals()},
        repeat=repeat, number=number)) / number
    real_mem = testing.last_compiled_executable.total_allocation_size()
    objective = testing.last_compiled_auto_sharding_objective

    # Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    assert_only_has_allreduce(hlo_ir)
    #print("===== HLO =====")
    #print(hlo_ir)

    #optimizer = closure[0]
    #sharding_specs = jax.tree_util.tree_map(lambda x: x.sharding_spec, optimizer)

    # Log benchmark results
    heads = ["Type", "Case", "PeakMem", "Objective", "Mean Time", "Std Time"]
    values = ["mlp", str(benchmark_case), f"{real_mem/GB:.2f}", f"{objective:.2f}",
             f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, "result_mlp.tsv")

    physical_mesh.shutdown()


benchmark_suite = [
    # Batch size, seq_len, hidden size, num_layers, dp_size, tensor_mp_size,
    (32,          1024,    2304,        4,          4,       1),
    (32,          1024,    2304,        4,          2,       2),

    # Batch size, seq_len, hidden size, num_layers, dp_size, tensor_mp_size,
    (8,           256,     5760,        4,          4,       1),
    (8,           256,     5760,        4,          2,       2),
]


def benchmark_all(use_profiling):
    for case in benchmark_suite:
        benchmark_mlp_one_case(case, use_profiling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--number", type=int, default=10)
    args = parser.parse_args()

    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)
