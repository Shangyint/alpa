"""Suites for wresnet benchmarking."""
from collections import namedtuple
import numpy as np

from benchmark_parallel_utils import (BenchmarkCase, SearchParallelArgs,
                                      LoadSolutionParallelArgs)

UNetModelConfig = namedtuple(
    "UNetModelConfig",
    ["image_size", "channel_size", "block_cnt", "dtype"])

unet_specs = {
    #
    "a": UNetModelConfig(384, 20, 2, np.float32),
    # "a": UNetModelConfig(384, 20, 4, np.float32),
    "b": UNetModelConfig(384, 40, 4, np.float32),
    "c": UNetModelConfig(384, 80, 4, np.float32),
    "d": UNetModelConfig(384, 160, 4, np.float32),
}

prefer_reduce_scatter = False
use_remat = True
force_batch_dim_mapping = False

auto_stage_option = {
    "submesh_physical_shape_space": "small_power_of_two",
    "submesh_logical_shape_space": "single_node_model_parallel",
    "stage_imbalance_tolerance": 0.25,
    "use_hlo_cost_model": False,
    "profiling_database_filename": None,
}

def get_num_auto_layers(name):
    return int(unet_specs[name].block_cnt * 1.5)

def get_search_cases(model_name, max_global_batch_size, num_micro_batches_list):
    num_auto_layers = get_num_auto_layers(model_name)
    return [
        BenchmarkCase(
            max_global_batch_size, unet_specs[model_name], num_micro_batches,
            "search",
            SearchParallelArgs(prefer_reduce_scatter, use_remat,
                               num_auto_layers, auto_stage_option))
        for num_micro_batches in num_micro_batches_list
    ]

def get_solution_case(model_name, max_global_batch_size, num_micro_batches,
                      forward_stage_layer_ids, submesh_physical_shapes,
                      submesh_logical_shapes,
                      submesh_autosharding_option_dicts):
    num_auto_layers = get_num_auto_layers(model_name)
    return [
        BenchmarkCase(
            max_global_batch_size, unet_specs[model_name], num_micro_batches,
            "load_solution",
            LoadSolutionParallelArgs(prefer_reduce_scatter, use_remat,
                                     num_auto_layers, forward_stage_layer_ids,
                                     submesh_physical_shapes,
                                     submesh_logical_shapes,
                                     submesh_autosharding_option_dicts))
    ]

# B = batch_size, I = image_size,
# L = num_layers, C = num_base_channels, W = width_factor,
# NB = num_micro_batches, PM = parallel_mode
# L_Shape = logical_mesh_shape
# RS = prefer_reduce_scatter, Remat = use_rematerialization,
# FM = force_batch_dim_mapping,

force_dp_dict = {"force_batch_dim_to_mesh_dim": 0}

# Performance test with shard parallel
tmp_suite = {}

# Performance test with shard parallel
# key = the number of gpus, value = a list of cases
# B,    I,   L,   C,   W, dtype,  NB, PM,          RS,    Remat, L_shape, FM
perf_test_2d_suite = {}

# Performance test with search solutions found for p3.16xlarge
perf_test_auto_suite = {}

# Grid search on hyperparameters
# key = the number of gpus, value = a list of cases
# model_name, B, NB
grid_search_auto_suite = {
    4: get_search_cases("a", 256, [16,])
}
