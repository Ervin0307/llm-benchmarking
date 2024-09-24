from typing import List
from random import randint
from itertools import product


class CollectivesInput:
    used_comm_ids = set()

    def __init__(
        self,
        num_workers: int,
        num_workers_per_node: int,
        collective_size: int,
        collective: str,
    ):
        self.num_workers = num_workers
        self.num_workers_per_node = num_workers_per_node
        self.collective_size = collective_size
        self.collective = collective
        self.comm_id = self._get_comm_id()

    @classmethod
    def _get_comm_id(cls):
        comm_id = randint(0, 65535)
        while comm_id in cls.used_comm_ids:
            comm_id = randint(0, 65535)
        cls.used_comm_ids.add(comm_id)
        return comm_id

    def is_valid(self, total_gpus_available: int, num_nodes: int):
        if self.num_workers == 1:
            return False

        if self.collective == "send_recv" and self.num_workers != 2:
            return False

        if (
            self.num_workers > self.num_workers_per_node
            and self.num_workers % self.num_workers_per_node != 0
        ):
            return False

        num_nodes_required = self.num_workers // self.num_workers_per_node

        if self.num_workers > total_gpus_available or num_nodes_required > num_nodes:
            return False

        return True


def get_collectives_sizes_to_profile(max_collective_size: int):
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 265 * 1024))
    )
    collectives_size_to_profile = []
    for collectives_size in COLLECTIVE_SIZE_SPACE:
        if collectives_size <= max_collective_size:
            collectives_size_to_profile.append(collectives_size)
        else:
            break
    return collectives_size_to_profile


def get_collectives_inputs(
    num_nodes: int,
    num_workers_per_node_combinations: List[int],
    max_collective_size: int,
    collective: str,
    total_gpus_available: int,
):
    num_workers = []

    for num_workers_per_node in num_workers_per_node_combinations:
        for _num_nodes in range(1, num_nodes + 1):
            num_workers.append(num_workers_per_node * _num_nodes)

    num_workers = list(set(num_workers))
    collectives_sizes = get_collectives_sizes_to_profile(max_collective_size)

    collectives_inputs = []

    for num_workers, num_workers_per_node, collective_size in product(
        num_workers, num_workers_per_node_combinations, collectives_sizes
    ):
        collectives_input = CollectivesInput(
            num_workers, num_workers_per_node, collective_size, collective
        )
        if not collectives_input.is_valid(total_gpus_available, num_nodes):
            continue

        collectives_inputs.append(collectives_input)

    return collectives_inputs
