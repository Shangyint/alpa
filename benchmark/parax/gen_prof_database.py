"""Generate the profiling result database."""
import ray
import argparse

import jax
from parax import DeviceCluster, ProfilingResultDatabase


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_key", type=str, default="default")
    parser.add_argument("--filename", type=str, default="prof_database.pkl")
    args = parser.parse_args()
    ray.init(address="auto")

    # Initialize a useless jax GPU backend in the driver script.
    # This GPU backend takes 300MB GPU memory to store the CUDA context.
    # This simulates the environment of our benchmark scripts and
    # can make the profiling of available memory more accurate.
    # TODO(lmzheng): Modify jax so it does not allocate this useless CUDA context.
    jax.config.update('jax_platform_name', 'cpu')
    _ = jax.numpy.ones(1)

    comm_size_range = (0, 29)
    cluster = DeviceCluster()
    prof_database = cluster.profile_all(args.cluster_key, comm_size_range=comm_size_range)
    prof_database.save(args.filename)
    print(f"Save profiling database to {args.filename}")