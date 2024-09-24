import gc
import os
from typing import Optional
import ray
import torch

from .collectives_input import CollectivesInput
from .collectives_wrapper import CollectiveWrapper
from llm_benchmark.utils.device_utils import get_numa_nodes


@ray.remote(num_cpus=1)
class BenchmarkRunner:
    def __init__(
        self, worker_id: int, workers_per_node: int, head_ip: str, device: str
    ) -> None:
        self._worker_id = worker_id
        self._max_workers_per_node = workers_per_node

        if device == "cuda":
            self._set_cuda_visible_devices()

        self._last_num_workers_per_node = None
        self._last_num_workers = None
        self._head_ip = head_ip
        self._device = device

    def _set_cuda_visible_devices(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            self._worker_id % self._max_workers_per_node
        )
        # set additional nccl env vars
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Required for properly capturing nccl ops
        os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
        os.environ["KINETO_LOG_LEVEL"] = "5"
        os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"

    def run_collective(
        self,
        collectives_input: CollectivesInput,
    ) -> Optional[dict]:
        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        rank = self._get_rank(
            collectives_input.num_workers, collectives_input.num_workers_per_node
        )
        if rank is None:
            return None

        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ):
            self._init_communication(
                collectives_input.comm_id,
                rank,
                collectives_input.num_workers,
                collectives_input.num_workers_per_node,
            )
            self._last_num_workers = collectives_input.num_workers
            self._last_num_workers_per_node = collectives_input.num_workers_per_node

        wrapper = CollectiveWrapper(
            rank,
            collectives_input.num_workers,
            collectives_input.comm_id,
            collectives_input.collective_size,
            collectives_input.collective,
            collectives_input.num_workers_per_node,
            self._max_workers_per_node,
            self._device,
        )
        stats = wrapper.profile()
        del wrapper
        gc.collect()
        return stats

    def _init_communication(
        self, comm_id: int, rank: int, num_workers: int, devices_per_node: int
    ):
        print(
            f"Initializing worker id: {self._worker_id}, Rank: {rank}, num_workers: {num_workers}, comm_id: {comm_id}, "
            f"devices_per_node: {devices_per_node}, max_workers_per_node: {self._max_workers_per_node}, "
            f"ip_addr: {ray.util.get_node_ip_address()}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        if self._device == "cuda":
            backend = "nccl"
        else:
            backend = "gloo"

        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=num_workers,
            init_method=f"tcp://{self._head_ip}:{comm_id}",
        )

    def _get_rank(self, num_workers: int, devices_per_node: int):
        assert self._max_workers_per_node >= devices_per_node
        assert self._max_workers_per_node % devices_per_node == 0
        assert num_workers % devices_per_node == 0 or num_workers < devices_per_node

        num_nodes = num_workers // devices_per_node
        current_node = self._worker_id // self._max_workers_per_node

        if current_node >= num_nodes:
            return None

        local_worker_id = self._worker_id % self._max_workers_per_node

        # # scatter devices uniformly across the node
        # node_devices = list(range(self._max_devices_per_node))
        # device_offset = self._max_devices_per_node // devices_per_node

        # # selected devices for this worker
        # selected_devices = node_devices[::device_offset]

        # pack devices in order
        selected_devices = list(range(devices_per_node))

        if local_worker_id not in selected_devices:
            return None

        # rank of this worker
        rank = current_node * devices_per_node + selected_devices.index(local_worker_id)

        return rank


def create_runner_pool(device: str = "cpu"):
    if device == "cpu":
        total_workers_available = len(get_numa_nodes())
    elif device in ["cuda", "gpu"]:
        total_workers_available = int(ray.cluster_resources()["GPU"])

    print(f"Total {device} available: {total_workers_available}")

    assert total_workers_available > 0, "No workers available"

    all_node_ips = [x["NodeName"] for x in ray.nodes()]
    print(f"All node IPs: {all_node_ips}")

    assert len(all_node_ips) > 0, "No nodes available"

    num_nodes = len(all_node_ips)
    workers_per_node = total_workers_available // len(all_node_ips)
    print(f"Workers per node: {workers_per_node}")
    runner_pool = []
    for worker_id in range(total_workers_available):
        node_ip = all_node_ips[worker_id // workers_per_node]
        runner_pool.append(
            BenchmarkRunner.options(
                num_cpus=1 if device == "cpu" else 0,
                num_gpus=1 if device == "cuda" else 0,
                resources={
                    f"node:{node_ip}": 0.01,
                },
            ).remote(worker_id, workers_per_node, all_node_ips[0], device)
        )
    return total_workers_available, num_nodes, runner_pool
