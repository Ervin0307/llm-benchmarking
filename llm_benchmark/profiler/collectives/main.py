import argparse
import datetime
import os
from typing import List
import pandas as pd
import ray
from tqdm import tqdm

from llm_benchmark.profiler.collectives.benchmark_runner import BenchmarkRunner
from llm_benchmark.profiler.utils import get_collectives_inputs


def get_numa_nodes():
    # On Linux systems, NUMA node information can be found under /sys/devices/system/node
    numa_nodes = []
    if os.path.exists('/sys/devices/system/node'):
        for node in os.listdir('/sys/devices/system/node'):
            if node.startswith('node'):
                numa_nodes.append(node)
    return numa_nodes

def create_runner_pool(device: str = "cpu"):
    if device == "cpu":
        total_workers_available = len(get_numa_nodes())
    elif device == "cuda":
        total_workers_available = int(ray.cluster_resources()[device.upper()])
        
    print(f"Total {device} available: {total_workers_available}")

    assert total_workers_available > 0, "No workers available"

    all_node_ips = [x["NodeName"] for x in ray.nodes()]
    print(f"All node IPs: {all_node_ips}")

    assert len(all_node_ips) > 0, "No nodes available"

    num_nodes = len(all_node_ips)
    workers_per_node = total_workers_available // len(all_node_ips)

    runner_pool = []
    for worker_id in range(total_workers_available):
        node_ip = all_node_ips[worker_id // workers_per_node]
        runner_pool.append(
            BenchmarkRunner.options(
                num_cpus=1 if device == "cpu" else 0,
                num_gpus=1 if device == "cuda" else 0,
                resources={
                    f"node:{node_ip}": 0.01,
                }
            ).remote(worker_id, workers_per_node, all_node_ips[0], device)
        )
    return total_workers_available, num_nodes, runner_pool


def profile_collectives(
        num_workers_per_node_combinations: List[int],
        max_collective_size: int,
        collective: str,
        device: str = "cpu", # "cpu" or "cuda"
        output_dir: str = "results",
    ):

    ray.init()

    total_workers_available, num_nodes, runner_pool = create_runner_pool(device)

    all_results = []

    collectives_inputs = get_collectives_inputs(
        num_nodes,
        num_workers_per_node_combinations,
        max_collective_size,
        collective,
        total_workers_available,
    )   

    for collectives_input in tqdm(collectives_inputs):
        promises = []
        for worker_id in range(total_workers_available):
            promise = runner_pool[worker_id].run_collective.remote(collectives_input)
            promises.append(promise)

        for worker_id in range(int(total_workers_available)):
            result = ray.get([promises[worker_id]])[0]
            if result and worker_id == 0:
                all_results.append(result)  

        ray.get(promises)

    # filter none results
    all_results = [x for x in all_results if x is not None]

    df = pd.DataFrame(all_results)
    # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix  
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    output_dir = f"{output_dir}/collective/"
    os.makedirs(output_dir, exist_ok=True)
    # write results to a csv file
    df.to_csv(f"{output_dir}/{collective}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    return df


if __name__ == "__main__":
    # main()
    profile_collectives(
        num_workers_per_node_combinations=[1, 2, 4],
        max_collective_size=512 * 1024,
        collective="all_reduce", # "all_reduce" or "send_recv"
        device="cpu", # "cpu" or "cuda" 
    )