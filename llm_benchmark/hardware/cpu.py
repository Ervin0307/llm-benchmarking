import psutil
import subprocess
from typing import Optional

from .constants import FlopsPerCycle


def get_flops_per_cycle(isa_info):
    flops_per_cycle = FlopsPerCycle.DEFAULT
    if isa_info.get("AMX"):
        # Estimate for AMX (assume 1024 FLOPs per cycle per core for AMX)
        flops_per_cycle = FlopsPerCycle.AMX
    elif isa_info.get("AVX512"):
        # Estimate for AVX-512 (assume 32 FLOPs per cycle per core for AVX-512)
        flops_per_cycle = FlopsPerCycle.AVX512
    elif isa_info.get("AVX2"):
        # Estimate for AVX2 (assume 16 FLOPs per cycle per core for AVX2)
        flops_per_cycle = FlopsPerCycle.AVX2
    elif isa_info.get("AVX"):
        # Estimate for AVX (assume 8 FLOPs per cycle per core for AVX)
        flops_per_cycle = FlopsPerCycle.AVX
    else:
        # Assume basic floating point performance (4 FLOPs per cycle per core)
        flops_per_cycle = FlopsPerCycle.DEFAULT

    return flops_per_cycle.value


def get_flops_per_second(
    clock_speed_mhz: float, num_cores: int, flops_per_cycle: float
) -> float:
    # Calculate the FLOPs
    flops = flops_per_cycle * clock_speed_mhz * num_cores * 1e6

    # Convert FLOPs to TFLOPs
    tflops = flops / 1e12

    return tflops


def get_cache_and_isa_info():
    lscpu_info = {}

    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        output = result.stdout

        lscpu_info["AVX"] = False
        lscpu_info["AMX"] = False

        for line in output.splitlines():
            if "L1d cache" in line:
                lscpu_info["l1_data_cache_size"] = line.split(":")[1].strip()
            if "L1i cache" in line:
                lscpu_info["l1_instruction_cache_size"] = line.split(":")[1].strip()
            if "L2 cache" in line:
                lscpu_info["l2_cache_size"] = line.split(":")[1].strip()
            if "L3 cache" in line:
                lscpu_info["l3_cache_size"] = line.split(":")[1].strip()
            if "Model name" in line:
                lscpu_info["model_name"] = line.split(":")[1].strip()

            if "Flags" in line or "flags" in line:
                flags = line.split(":")[1].strip().split()
                if "avx" in flags or "avx2" in flags or "avx512f" in flags:
                    lscpu_info["AVX"] = True
                if "amx_bf16" in flags or "amx_tile" in flags or "amx_int8" in flags:
                    lscpu_info["AMX"] = True
    except Exception as e:
        print(f"CPU extraction failed with {str(e)}")
    finally:
        return lscpu_info


def get_cores_and_mem_info(pid: Optional[int] = None, current_only: bool = False):
    cm_info = {}

    if pid is not None:
        try:
            proc = psutil.Process(pid)

            cm_info["cpu_utilization"] = proc.cpu_percent(
                interval=None if current_only else 1.0
            )
            per_core_util = psutil.cpu_percent(interval=None, percpu=True)
            cpu_core_affinity = proc.cpu_affinity()
            cm_info["cpu_core_affinity"] = ",".join(map(str, cpu_core_affinity))
            cm_info["cpu_utilization_per_core"] = ",".join(
                map(str, [per_core_util[core] for core in cpu_core_affinity])
            )

            cpu_freq = psutil.cpu_freq(percpu=True)
            cm_info["cpu_freq_current_per_core"] = ",".join(
                map(str, [cpu_freq[core].current for core in cpu_core_affinity])
            )

            cm_info["cpu_memory_used"] = proc.memory_info().rss
            cm_info["cpu_memory_utilization"] = proc.memory_percent()
            mem_info = psutil.virtual_memory()
        except psutil.NoSuchProcess as e:
            print(f"CPU extraction failed with {str(e)}")
            return cm_info
    else:
        cm_info["cpu_utilization"] = psutil.cpu_percent(
            interval=None if current_only else 1.0
        )

        mem_info = psutil.virtual_memory()
        cm_info["cpu_memory_used"] = mem_info.used
        cm_info["cpu_memory_utilization"] = mem_info.percent

    cm_info["cpu_memory_total"] = mem_info.total
    cm_info["cpu_memory_available"] = mem_info.available

    cpu_freq = psutil.cpu_freq()
    avg_load = psutil.getloadavg()

    cm_info.update(
        {
            "cpu_freq_current": cpu_freq.current,
            "cpu_load_1min": avg_load[0],
            "cpu_load_5min": avg_load[1],
            "cpu_load_15min": avg_load[2],
        }
    )

    if current_only:
        return cm_info

    num_physical_cores = psutil.cpu_count(logical=False)
    num_virtual_cores = psutil.cpu_count(logical=True)

    cm_info.update(
        {
            "cpu_freq_min": cpu_freq.min,
            "cpu_freq_max": cpu_freq.max,
            "num_physical_cores": num_physical_cores,
            "num_virtual_cores": num_virtual_cores,
            "threads_per_core": num_virtual_cores / num_physical_cores,
        }
    )

    return cm_info


def get_temp_and_power_info(current_only: bool = False):
    temps = psutil.sensors_temperatures()
    core_temp = temps.get("coretemp", [])
    avg_temp_current = sum(int(temp.current) for temp in core_temp) / max(
        1, len(core_temp)
    )

    tp_info = {"cpu_temp_current": avg_temp_current}

    if current_only:
        return tp_info

    tp_info["cpu_temp_high"] = sum(int(temp.high) for temp in core_temp) / max(
        1, len(core_temp)
    )

    return tp_info


def get_cpu_info():
    info = get_cores_and_mem_info()
    info.update(get_temp_and_power_info())

    cache_isa_info = get_cache_and_isa_info()

    info["flops_per_cycle"] = get_flops_per_cycle(cache_isa_info)
    info["tflops_current"] = get_flops_per_second(
        info["cpu_freq_current"], info["num_physical_cores"], info["flops_per_cycle"]
    )
    info["tflops_min"] = get_flops_per_second(
        info["cpu_freq_min"], info["num_physical_cores"], info["flops_per_cycle"]
    )
    info["tflops_max"] = get_flops_per_second(
        info["cpu_freq_max"], info["num_physical_cores"], info["flops_per_cycle"]
    )

    return {**info, **cache_isa_info}
