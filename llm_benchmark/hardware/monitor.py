import os
import csv
import time
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from . import cuda as cuda_utils
from . import cpu as cpu_utils
from llm_benchmark.utils.device_utils import get_available_devices


def get_cpu_usage(pid: int = None):
    info = cpu_utils.get_cores_and_mem_info(pid, current_only=True)
    info.update(cpu_utils.get_temp_and_power_info(current_only=True))
    return info


def get_gpu_usage(pid: int = None):
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_metrics = {}

    for idx in range(num_gpus):
        info = cuda_utils.get_cores_info(idx, current_only=True)
        info.update(cuda_utils.get_memory_info(idx))
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
        import pynvml
        pynvml.nvmlInit()

    try:
        while (end_time is None or time.time() < end_time) and (
            stop_event is None or not stop_event.is_set()
        ):
            if isinstance(metadata, dict):
                metrics = deepcopy(metadata)
            else:
                metrics = {}

            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_usage_info = get_cpu_usage(pid) or {}
            if not len(cpu_usage_info):
                break

            metrics.update(cpu_usage_info)
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
