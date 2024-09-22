import psutil
import subprocess

from .constants import FlopsPerCycle

def calculate_flops_per_cycle(isa_info):
    flops_per_cycle = FlopsPerCycle.DEFAULT
    if isa_info['AMX']:
        # Estimate for AMX (assume 1024 FLOPs per cycle per core for AMX)
        flops_per_cycle = FlopsPerCycle.AMX
    elif isa_info['AVX512']:
        # Estimate for AVX-512 (assume 32 FLOPs per cycle per core for AVX-512)
        flops_per_cycle = FlopsPerCycle.AVX512
    elif isa_info['AVX2']:
        # Estimate for AVX2 (assume 16 FLOPs per cycle per core for AVX2)
        flops_per_cycle = FlopsPerCycle.AVX2
    elif isa_info['AVX']:
        # Estimate for AVX (assume 8 FLOPs per cycle per core for AVX)
        flops_per_cycle = FlopsPerCycle.AVX
    else:
        # Assume basic floating point performance (4 FLOPs per cycle per core)
        flops_per_cycle = FlopsPerCycle.DEFAULT

    return flops_per_cycle

def get_flops_per_second(clock_speed_mhz: float, num_cores: int, flops_per_cycle: float) -> float:    
    # Calculate the FLOPs
    flops = flops_per_cycle * clock_speed_mhz * num_cores * 1e6
    
    # Convert FLOPs to TFLOPs
    tflops = flops / 1e12
    
    return tflops

def get_lscpu_info():
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        output = result.stdout
        

        lscpu_info = {}
        lscpu_info['AVX'] = False
        lscpu_info['AMX'] = False

        for line in output.splitlines():
            if "L1d cache" in line:
                lscpu_info['L1d'] = line.split(':')[1].strip()
            if "L1i cache" in line:
                lscpu_info['L1i'] = line.split(':')[1].strip()
            if "L2 cache" in line:
                lscpu_info['L2'] = line.split(':')[1].strip()
            if "L3 cache" in line:
                lscpu_info['L3'] = line.split(':')[1].strip()
            if "Model name" in line:
                lscpu_info['model_name'] = line.split(':')[1].strip()
            
            if "Flags" in line or "flags" in line:
                flags = line.split(':')[1].strip().split()
                if 'avx' in flags or 'avx2' in flags or 'avx512f' in flags:
                    lscpu_info['AVX'] = True
                if 'amx_bf16' in flags or 'amx_tile' in flags or 'amx_int8' in flags:
                    lscpu_info['AMX'] = True

        return lscpu_info

    except Exception as e:
        return str(e)

def get_cpu_info():

    info = {}
    
    info['cpu_percentage'] = psutil.cpu_percent(interval=1)
    # Get number of cores
    info["num_physical_cores"] = psutil.cpu_count(logical=False)
    info["num_virtual_cores"] = psutil.cpu_count(logical=True)
    info["threads_per_core"] = info["num_virtual_cores"] / info["num_physical_cores"]
    # Get CPU frequency
    cpu_freq = psutil.cpu_freq()
    info["cpu_freq_current"] = cpu_freq[0]
    info["cpu_freq_min"] = cpu_freq[1]
    info["cpu_freq_max"] = cpu_freq[2]

    avg_load = psutil.getloadavg()
    info["cpu_load_1min"] = avg_load[0]
    info["cpu_load_5min"] = avg_load[1]
    info["cpu_load_15min"] = avg_load[2]

    # Get CPU memory
    cpu_memory = psutil.virtual_memory()
    info["cpu_memory_total"] = cpu_memory[0]
    info["cpu_memory_available"] = cpu_memory[1]
    info["cpu_memory_used"] = cpu_memory[2]
    info["cpu_memory_percentage"] = cpu_memory[3]

    temps = psutil.sensors_temperatures()
    avg_temp_current = sum(int(temp.current) for temp in temps['coretemp']) / len(temps['coretemp'])
    avg_temp_high = sum(int(temp.high) for temp in temps['coretemp']) / len(temps['coretemp'])
    info["cpu_temp_current"] = avg_temp_current
    info["cpu_temp_high"] = avg_temp_high


    lscpu_info = get_lscpu_info()

    info['flops_per_cycle'] = calculate_flops_per_cycle(lscpu_info)
    info['tflops_current'] = get_flops_per_second(info['cpu_freq_current'], info['num_physical_cores'], info['flops_per_cycle'])
    info['tflops_min'] = get_flops_per_second(info['cpu_freq_min'], info['num_physical_cores'], info['flops_per_cycle'])
    info['tflops_max'] = get_flops_per_second(info['cpu_freq_max'], info['num_physical_cores'], info['flops_per_cycle'])
    
    return {**info, **lscpu_info}

print(get_cpu_info())