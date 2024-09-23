import time
import json
import psutil
import pynvml
import docker
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime


client = docker.from_env()
pynvml.nvmlInit()


async def write_metrics_to_file(log_data, output_file):
    with open(output_file, "a") as f:
        f.write("\n".join(log_data) + "\n")


def get_container_stats(container_id: str):
    container = client.containers.get(container_id)
    stats = container.stats(stream=False)

    cpu_delta = (
        stats["cpu_stats"]["cpu_usage"]["total_usage"]
        - stats["precpu_stats"]["cpu_usage"]["total_usage"]
    )
    system_delta = (
        stats["cpu_stats"]["system_cpu_usage"]
        - stats["precpu_stats"]["system_cpu_usage"]
    )
    cpu_usage_percent = (
        (cpu_delta / system_delta)
        * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"])
        * 100
    )

    mem_usage = stats["memory_stats"]["usage"]
    mem_limit = stats["memory_stats"]["limit"]
    mem_percent = (mem_usage / mem_limit) * 100

    gpu_stats = stats.get("gpu_stats", {})

    return {
        "cpu_percent": cpu_usage_percent,
        "memory_usage": mem_usage,
        "memory_limit": mem_limit,
        "memory_percent": mem_percent,
        "gpu_stats": gpu_stats,
    }


def get_cpu_memory_usage(pid: int = None):
    if pid is None:
        cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
        memory_info = psutil.virtual_memory()

        return {
            "cpu_percent_total": psutil.cpu_percent(interval=None),
            "cpu_percent_per_core": cpu_percent_per_core,
            "memory_total": memory_info.total,
            "memory_used": memory_info.used,
            "memory_percent": memory_info.percent,
        }
    else:
        try:
            proc = psutil.Process(pid)
            # Process-specific CPU and memory stats
            cpu_percent = proc.cpu_percent(interval=None)
            mem_info = proc.memory_info()
            core_affinity = proc.cpu_affinity()

            cpu_usage_per_core = {
                f"core_{core}": psutil.cpu_percent(interval=None, percpu=True)[core]
                for core in core_affinity
            }

            return {
                "cpu_percent": cpu_percent,
                "cpu_affinity_cores": core_affinity,
                "cpu_usage_per_affinity_core": cpu_usage_per_core,
                "memory_used": mem_info.rss,  # Resident Set Size (physical memory usage)
                "memory_percent": proc.memory_percent(),
                "num_threads": proc.num_threads(),
            }
        except psutil.NoSuchProcess:
            return


def get_gpu_usage(pid: int = None):
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_metrics = {}

    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_metrics[f"gpu_{i}"] = {
            "gpu_percent": gpu_util.gpu,
            "gpu_memory_total": mem_info.total,
            "gpu_memory_used": mem_info.used,
            "gpu_memory_percent": (mem_info.used / mem_info.total) * 100,
        }

    return gpu_metrics


async def log_system_metrics(
    output_path: str, pid: int = None, interval: int = 5, duration: int = None
):
    log_data = []
    end_time = (time.time() + duration) if duration is not None else None
    print(
        f"Logging system metrics (pid={pid}) every {interval} seconds for {duration} seconds..."
    )

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    output_file = Path(output_path, "system_metrics.jsonl")

    try:
        while end_time is None or time.time() < end_time:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            cpu_memory_metrics = get_cpu_memory_usage(pid) or {}

            gpu_metrics = get_gpu_usage(pid) or {}

            log_data.append(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "cpu_memory_metrics": cpu_memory_metrics,
                        "gpu_metrics": gpu_metrics,
                    },
                    indent=4,
                )
            )

            if len(log_data) >= 100:
                await write_metrics_to_file(log_data, output_file)
                log_data.clear()

            await asyncio.sleep(interval)
    except Exception as e:
        print(f"Monitoring interrupted with exception, {e}")
    finally:
        if len(log_data):
            await write_metrics_to_file(log_data, output_file)

        print(f"Metrics logged to {output_file}")


if __name__ == "__main__":
    asyncio.run(log_system_metrics(output_path=".", pid=1508558, interval=5))
