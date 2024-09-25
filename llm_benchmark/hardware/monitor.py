import os
import csv
import time
import pynvml
import psutil
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from llm_benchmark.hardware import cuda as cuda_utils
from llm_benchmark.utils.device_utils import get_available_devices


def get_cpu_memory_usage(pid: int = None):
    if pid is None:
        cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
        memory_info = psutil.virtual_memory()

        return {
            "cpu_utilization": psutil.cpu_percent(interval=None),
            "cpu_utilization_per_core": ",".join(map(str, cpu_percent_per_core)),
            "cpu_memory_total": memory_info.total,
            "cpu_memory_used": memory_info.used,
            "cpu_memory_percent": memory_info.percent,
        }
    else:
        try:
            proc = psutil.Process(pid)
            # Process-specific CPU and memory stats
            cpu_percent = proc.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            core_affinity = proc.cpu_affinity()

            cpu_usage_per_core = [
                psutil.cpu_percent(interval=None, percpu=True)[core]
                for core in core_affinity
            ]

            return {
                "cpu_utilization": cpu_percent,
                "cpu_affinity_cores": core_affinity,
                "cpu_utilization_per_core": ",".join(map(str, cpu_usage_per_core)),
                "cpu_memory_used": mem_info.rss,  # Resident Set Size (physical memory usage)
                "cpu_memory_percent": proc.memory_percent(),
                "cpu_num_threads": proc.num_threads(),
            }
        except psutil.NoSuchProcess:
            return


def get_gpu_usage(pid: int = None):
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_metrics = {}

    for idx in range(num_gpus):
        info = cuda_utils.get_memory_info(idx)
        info.update(cuda_utils.get_clock_info(idx, current_only=True))
        info.update(cuda_utils.get_temp_and_power_info(idx, current_only=True))
        info.update(cuda_utils.get_pci_info(idx, current_only=True))
        info["throttle_reason"] = cuda_utils.get_throttle_reasons(idx)

        for key in info:
            if key not in gpu_metrics:
                gpu_metrics[key] = []
            gpu_metrics[key].append(info[key])

    return {metric: ",".join(map(str, gpu_metrics[metric])) for metric in gpu_metrics}


def log_system_metrics(
    output_dir: str,
    pid: int = None,
    interval: int = 5,
    duration: int = None,
    stop_event=None,
    metadata: dict = None,
):
    end_time = (time.time() + duration) if duration is not None else None
    print(
        f"Logging system metrics (pid={pid}) every {interval} seconds for {duration} seconds..."
    )

    Path(output_dir).parent.mkdir(exist_ok=True, parents=True)
    output_file = Path(output_dir, "system_metrics.csv")

    available_devices = get_available_devices()
    is_gpu_available = "cuda" in available_devices or "gpu" in available_devices
    if is_gpu_available:
        pynvml.nvmlInit()

    try:
        while (end_time is None or time.time() < end_time) or (
            stop_event is None or not stop_event.is_set()
        ):
            if isinstance(metadata, dict):
                metrics = deepcopy(metadata)
            else:
                metrics = {}

            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics.update(get_cpu_memory_usage(pid) or {})

            if is_gpu_available:
                metrics.update(get_gpu_usage(pid) or {})

            file_exists = os.path.isfile(output_file)
            with open(output_file, "a") as fp:
                fieldnames = list(metrics.keys())
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()

                writer.writerow(metrics)

            time.sleep(interval)
    except Exception as e:
        raise RuntimeError(f"Monitoring interrupted with exception, {e}")
    finally:
        if is_gpu_available:
            pynvml.nvmlShutdown()

        print(f"Metrics logged to {output_file}")
