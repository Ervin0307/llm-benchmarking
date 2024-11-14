import os
import math
import torch


def get_available_devices():
    devices = ["cpu"]

    if os.system("lspci | grep -i nvidia > /dev/null") == 0 or torch.cuda.is_available():
        devices.append("cuda")

    if os.system("lspci | grep -i habana > /dev/null") == 0:
        devices.append("hpu")

    return devices


def get_tensor_parallel_sizes(device_count):
    # Powers of 2 up to the maximum device count
    powers_of_2 = [
        2**i for i in range(int(math.log2(device_count)) + 1) if 2**i <= device_count
    ]

    # Factors of the device count
    factors = [i for i in range(1, device_count + 1) if device_count % i == 0]

    # Combine powers of 2 and factors, removing duplicates
    combined_sizes = sorted(set(powers_of_2 + factors))

    return combined_sizes


def get_numa_nodes():
    # On Linux systems, NUMA node information can be found under /sys/devices/system/node
    numa_nodes = []
    if os.path.exists("/sys/devices/system/node"):
        for node in os.listdir("/sys/devices/system/node"):
            if node.startswith("node"):
                numa_nodes.append(node)
    return numa_nodes
