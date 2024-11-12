import subprocess

from llm_benchmark.hardware.constants import DeviceInfo

def get_hpu_info(current_only=False):
    hpu_info = []

    try:
        # Fetch product name and memory information using hl-smi
        output = subprocess.check_output("hl-smi --query-aip=name,serial,memory.total,memory.free,memory.used,power.draw,temperature.aip,driver_version,clocks.current.soc,clocks.max.soc --format=csv,nounits", shell=True).decode().strip()
        lines = output.splitlines()
        
        # Skip the header line and process each device
        for line in lines[1:]:
            product_name, serial, memory_total, memory_free, memory_used, power_draw, temperature_aip, driver_version, clocks_current_soc, clocks_max_soc = line.split(',')
            device_info = {
                "memory_free": int(memory_free.strip()) / (1024),
                "memory_used": int(memory_used.strip()) / (1024),
                "power": int(power_draw.strip()),
                "temperature": int(temperature_aip.strip()),
                "clocks_current_soc": clocks_current_soc.strip(),
                "clocks_max_soc": clocks_max_soc.strip()
            }
            if not current_only:
                device_info["product_name"] = product_name.upper().replace("-", "_")
                device_info["serial"] = serial
                device_info["driver_version"] = driver_version.strip()
                device_info["memory"] = int(memory_total.strip()) / (1024)

            hpu_info.append(device_info)
    except subprocess.CalledProcessError as e:  
        print(f"Error fetching HPU info: {e}")

    return hpu_info
    
def create_hpu_config():
    
    dev_info = get_hpu_info()

    device_info = DeviceInfo['GAUDI2'].value
    device_config = {
        "name": dev_info[0]['product_name'],
        "mem_per_GPU_in_GB": device_info.mem_per_GPU_in_GB,
        "hbm_bandwidth_in_GB_per_sec": device_info.hbm_bandwidth_in_GB_per_sec,
        "intra_node_bandwidth_in_GB_per_sec": device_info.intra_node_bandwidth_in_GB_per_sec,
        "intra_node_min_message_latency": device_info.intra_node_min_message_latency,
        "peak_fp16_TFLOPS": device_info.peak_fp16_TFLOPS,
        "peak_i8_TFLOPS": device_info.peak_i8_TFLOPS,
        "peak_i4_TFLOPS": device_info.peak_i4_TFLOPS,
        "inter_node_bandwidth_in_GB_per_sec": device_info.inter_node_bandwidth_in_GB_per_sec,
        "available_count": len(dev_info)
    }
    return device_config, dev_info