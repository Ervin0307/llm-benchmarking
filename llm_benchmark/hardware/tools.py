import os
import csv
import torch
import datetime

from .benchmark import flops_benchmark, memory_bandwidth_benchmark
from .cpu import get_cpu_info

def get_hardware_info(cpu_only=False, output_dir=None):

    cpu_info = get_cpu_info()
    device = torch.device('cpu' if cpu_only else 'cuda')

    actual_flops = flops_benchmark(device)
    cpu_info['actual_flops'] = actual_flops
    cpu_info['memory_bandwidth'] = memory_bandwidth_benchmark(device)
    
    if output_dir:
        output_dir = f"{output_dir}/hardware/"
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        csv_file_path = os.path.join(output_dir, f"hardware_info_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

        file_exists = os.path.isfile(csv_file_path)
        # Open the CSV file in append mode
        with open(csv_file_path, "a", newline="") as csvfile:
            fieldnames = list(cpu_info.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write the summary data
            writer.writerow(cpu_info)

        print(f"Hardware info saved to {output_dir}")

    return cpu_info
