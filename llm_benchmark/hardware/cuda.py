import re
import subprocess

import pycuda.driver as cuda
import pycuda.autoinit


def get_gpu_specs_pycuda():
    """Uses pycuda to extract additional GPU specs like CUDA cores and SM count."""
    num_gpus = cuda.Device.count()
    gpu_specs = []

    for gpu_id in range(num_gpus):
        device = cuda.Device(gpu_id)
        attributes = device.get_attributes()

        # CUDA cores and other properties depend on the number of SMs (Streaming Multiprocessors)
        sm_count = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

        gpu_info = {
            "sm_count": sm_count,
            "compute_capability": device.compute_capability(),
        }

        for key in (
            "MAX_THREADS_PER_BLOCK",
            "MAX_THREADS_PER_MULTIPROCESSOR",
            "MAX_BLOCKS_PER_MULTIPROCESSOR",
            "TOTAL_CONSTANT_MEMORY",
            "GPU_OVERLAP",
            "SHARED_MEMORY_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK",
            "MAX_SHARED_MEMORY_PER_BLOCK_OPTIN",
            "MAX_SHARED_MEMORY_PER_MULTIPROCESSOR",
            "REGISTERS_PER_BLOCK",
            "MAX_REGISTERS_PER_BLOCK",
            "MAX_REGISTERS_PER_MULTIPROCESSOR",
            "INTEGRATED",
            "CAN_MAP_HOST_MEMORY",
            "COMPUTE_MODE",
            "CONCURRENT_KERNELS",
            "ECC_ENABLED",
            "PCI_BUS_ID",
            "PCI_DEVICE_ID",
            "TCC_DRIVER",
            "GLOBAL_MEMORY_BUS_WIDTH",
            "L2_CACHE_SIZE",
            "ASYNC_ENGINE_COUNT",
            "PCI_DOMAIN_ID",
            "STREAM_PRIORITIES_SUPPORTED",
            "GLOBAL_L1_CACHE_SUPPORTED",
            "LOCAL_L1_CACHE_SUPPORTED",
            "MANAGED_MEMORY",
            "MULTI_GPU_BOARD",
            "MULTI_GPU_BOARD_GROUP_ID",
            "SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO",
            "PAGEABLE_MEMORY_ACCESS",
            "CONCURRENT_MANAGED_ACCESS",
            "COMPUTE_PREEMPTION_SUPPORTED",
            "MAX_PERSISTING_L2_CACHE_SIZE",
        ):
            try:
                gpu_info[key.lower()] = attributes[getattr(cuda.device_attribute, key)]
            except AttributeError:
                gpu_info[key.lower()] = None

        gpu_specs.append(gpu_info)

    return gpu_specs


def get_gpu_specs_smi():
    gpu_specs = {}

    try:

        def extract_value(pattern, text, default="N/A"):
            match = re.search(pattern, text)
            return match.group(1).strip() if match else default

        def parse_pci_info(nvidia_smi_output):
            pci_info = {
                "pcie_generation_max": None,
                "pcie_generation_current": None,
                "pcie_device_current": None,
                "pcie_device_max": None,
                "pcie_host_max": None,
                "link_width_max": None,
                "link_width_current": None,
                "tx_throughput": None,
                "rx_throughput": None,
            }

            lines = nvidia_smi_output.splitlines()
            capturing = False
            for i, line in enumerate(lines):
                line = line.strip()

                # Start capturing when PCIe Generation is found
                if line.startswith("PCIe Generation"):
                    capturing = True
                    continue

                if capturing:
                    if line == "":
                        break

                    # Capture PCIe Generation details
                    match = re.match(r"\s*Max\s*:\s*(\d+)", line)
                    if match and pci_info["pcie_generation_max"] is None:
                        pci_info["pcie_generation_max"] = match.group(1)

                    match = re.match(r"\s*Current\s*:\s*(\d+)", line)
                    if match and pci_info["pcie_generation_current"] is None:
                        pci_info["pcie_generation_current"] = match.group(1)

                    match = re.match(r"\s*Device Current\s*:\s*(\d+)", line)
                    if match and pci_info["pcie_device_current"] is None:
                        pci_info["pcie_device_current"] = match.group(1)

                    match = re.match(r"\s*Device Max\s*:\s*(\d+)", line)
                    if match and pci_info["pcie_device_max"] is None:
                        pci_info["pcie_device_max"] = match.group(1)

                    match = re.match(r"\s*Host Max\s*:\s*(N/A|\d+)", line)
                    if match and pci_info["pcie_host_max"] is None:
                        pci_info["pcie_host_max"] = match.group(1)

                    # Check for Link Width and Throughput details
                    if line.startswith("Link Width"):
                        link_width_max = lines[i + 1].strip()
                        link_width_current = lines[i + 2].strip()

                        match = re.match(r"\s*Max\s*:\s*(\d+x)", link_width_max)
                        if match and pci_info["link_width_max"] is None:
                            pci_info["link_width_max"] = match.group(1)

                        match = re.match(r"\s*Current\s*:\s*(\d+x)", link_width_current)
                        if match and pci_info["link_width_current"] is None:
                            pci_info["link_width_current"] = match.group(1)

                    # Capture Throughput details
                    if line.startswith("Tx Throughput"):
                        match = re.match(r"Tx Throughput\s*:\s*(\d+\s*KB/s)", line)
                        if match and pci_info["tx_throughput"] is None:
                            pci_info["tx_throughput"] = match.group(1)

                    if line.startswith("Rx Throughput"):
                        match = re.match(r"Rx Throughput\s*:\s*(\d+\s*KB/s)", line)
                        if match and pci_info["rx_throughput"] is None:
                            pci_info["rx_throughput"] = match.group(1)

            return pci_info

        def parse_clocks_event_reasons(nvidia_smi_output):
            clocks_info = {
                "event_idle": None,
                "event_applications_clocks_setting": None,
                "event_sw_power_cap": None,
                "event_hw_slowdown": None,
                "event_hw_thermal_slowdown": None,
                "event_sw_thermal_slowdown": None,
                "event_hw_power_brake_slowdown": None,
                "event_sync_boost": None,
                "event_display_clock_setting": None,
            }

            lines = nvidia_smi_output.splitlines()
            capturing = False
            for i, line in enumerate(lines):
                line = line.strip()

                # Start capturing when "Clocks Event Reasons" is found
                if line.startswith("Clocks Event Reasons"):
                    capturing = True
                    continue

                if capturing:
                    # Stop capturing when we reach a line that doesn't belong
                    if line == "":
                        break

                    # Capture each event reason status
                    match = re.match(r"Idle\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_idle"] = match.group(1).strip()

                    match = re.match(r"Applications Clocks Setting\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_applications_clocks_setting"] = match.group(
                            1
                        ).strip()

                    match = re.match(r"SW Power Cap\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_sw_power_cap"] = match.group(1).strip()

                    match = re.match(r"HW Slowdown\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_hw_slowdown"] = match.group(1).strip()

                    if line.startswith("HW Thermal Slowdown"):
                        match = re.match(r"\s*HW Thermal Slowdown\s*:\s*(.*)", line)
                        if match:
                            clocks_info["event_hw_thermal_slowdown"] = match.group(
                                1
                            ).strip()

                    match = re.match(r"SW Thermal Slowdown\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_sw_thermal_slowdown"] = match.group(
                            1
                        ).strip()

                    if line.startswith("HW Power Brake Slowdown"):
                        match = re.match(r"\s*HW Power Brake Slowdown\s*:\s*(.*)", line)
                        if match:
                            clocks_info["event_hw_power_brake_slowdown"] = match.group(
                                1
                            ).strip()

                    match = re.match(r"Sync Boost\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_sync_boost"] = match.group(1).strip()

                    match = re.match(r"Display Clock Setting\s*:\s*(.*)", line)
                    if match:
                        clocks_info["event_display_clock_setting"] = match.group(
                            1
                        ).strip()

            return clocks_info

        def parse_memory_usage(nvidia_smi_output):
            memory_info = {
                "fb_memory_total": None,
                "fb_memory_reserved": None,
                "fb_memory_used": None,
                "fb_memory_free": None,
                "bar1_memory_total": None,
                "bar1_memory_used": None,
                "bar1_memory_free": None,
                "conf_memory_total": None,
                "conf_memory_used": None,
                "conf_memory_free": None,
            }

            lines = nvidia_smi_output.splitlines()
            capturing_fb = False
            capturing_bar1 = False
            capturing_conf = False
            for line in lines:
                line = line.strip()

                # Start capturing FB Memory Usage
                if line.startswith("FB Memory Usage"):
                    capturing_fb = True
                    capturing_bar1 = False
                    capturing_conf = False
                    continue

                # Start capturing BAR1 Memory Usage
                if line.startswith("BAR1 Memory Usage"):
                    capturing_bar1 = True
                    capturing_fb = False
                    capturing_conf = False
                    continue

                # Start capturing Conf Compute Protected Memory Usage
                if line.startswith("Conf Compute Protected Memory Usage"):
                    capturing_conf = True
                    capturing_fb = False
                    capturing_bar1 = False
                    continue

                if capturing_fb:
                    if "Total" in line:
                        match = re.match(r"Total\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["fb_memory_total"] = match.group(1).strip()
                    elif "Reserved" in line:
                        match = re.match(r"Reserved\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["fb_memory_reserved"] = match.group(1).strip()
                    elif "Used" in line:
                        match = re.match(r"Used\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["fb_memory_used"] = match.group(1).strip()
                    elif "Free" in line:
                        match = re.match(r"Free\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["fb_memory_free"] = match.group(1).strip()

                elif capturing_bar1:
                    if "Total" in line:
                        match = re.match(r"Total\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["bar1_memory_total"] = match.group(1).strip()
                    elif "Used" in line:
                        match = re.match(r"Used\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["bar1_memory_used"] = match.group(1).strip()
                    elif "Free" in line:
                        match = re.match(r"Free\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["bar1_memory_free"] = match.group(1).strip()

                elif capturing_conf:
                    if "Total" in line:
                        match = re.match(r"Total\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["conf_memory_total"] = match.group(1).strip()
                    elif "Used" in line:
                        match = re.match(r"Used\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["conf_memory_used"] = match.group(1).strip()
                    elif "Free" in line:
                        match = re.match(r"Free\s*:\s*(\d+)\s*MiB", line)
                        if match:
                            memory_info["conf_memory_free"] = match.group(1).strip()

            return memory_info

        def parse_temperature_and_power(nvidia_smi_output):
            power_info = {
                "gpu_current_temp": None,
                "memory_current_temp": None,
                "gpu_max_operating_temp": None,
                "gpu_shutdown_temp": None,
                "gpu_slowdown_temp": None,
                "gpu_t_limit_temp": None,
                "memory_max_operating_temp": None,
                "power_draw": None,
                "current_power_limit": None,
                "requested_power_limit": None,
                "default_power_limit": None,
                "min_power_limit": None,
                "max_power_limit": None,
            }

            lines = nvidia_smi_output.splitlines()
            capturing_temperature = False
            capturing_power = False
            for line in lines:
                line = line.strip()

                # Start capturing Temperature readings
                if line.startswith("Temperature"):
                    capturing_temperature = True
                    capturing_power = False
                    continue

                # Start capturing Power readings
                if line.startswith("GPU Power Readings"):
                    capturing_power = True
                    capturing_temperature = False
                    continue

                if capturing_temperature:
                    if "GPU Current Temp" in line:
                        match = re.match(r"GPU Current Temp\s*:\s*(\d+)\s*C", line)
                        if match:
                            power_info["gpu_current_temp"] = match.group(1).strip()
                    elif "Memory Current Temp" in line:
                        match = re.match(r"Memory Current Temp\s*:\s*(\d+)\s*C", line)
                        if match:
                            power_info["memory_current_temp"] = match.group(1).strip()
                    elif "GPU Max Operating Temp" in line:
                        match = re.match(
                            r"GPU Max Operating Temp\s*:\s*(\d+)\s*C", line
                        )
                        if match:
                            power_info["gpu_max_operating_temp"] = match.group(
                                1
                            ).strip()
                    elif "GPU Shutdown Temp" in line:
                        match = re.match(r"GPU Shutdown Temp\s*:\s*(\d+)\s*C", line)
                        if match:
                            power_info["gpu_shutdown_temp"] = match.group(1).strip()
                    elif "GPU Slowdown Temp" in line:
                        match = re.match(r"GPU Slowdown Temp\s*:\s*(\d+)\s*C", line)
                        if match:
                            power_info["gpu_slowdown_temp"] = match.group(1).strip()
                    elif "GPU T.Limit Temp" in line:
                        match = re.match(r"GPU T\.Limit Temp\s*:\s*(N/A|\d+)\s*C", line)
                        if match:
                            power_info["gpu_t_limit_temp"] = match.group(1).strip()
                    elif "Memory Max Operating Temp" in line:
                        match = re.match(
                            r"Memory Max Operating Temp\s*:\s*(\d+)\s*C", line
                        )
                        if match:
                            power_info["memory_max_operating_temp"] = match.group(
                                1
                            ).strip()

                elif capturing_power:
                    if "Power Draw" in line:
                        match = re.match(r"Power Draw\s*:\s*([\d.]+)\s*W", line)
                        if match:
                            power_info["power_draw"] = match.group(1).strip()
                    elif "Current Power Limit" in line:
                        match = re.match(
                            r"Current Power Limit\s*:\s*([\d.]+)\s*W", line
                        )
                        if match:
                            power_info["current_power_limit"] = match.group(1).strip()
                    elif "Requested Power Limit" in line:
                        match = re.match(
                            r"Requested Power Limit\s*:\s*([\d.]+)\s*W", line
                        )
                        if match:
                            power_info["requested_power_limit"] = match.group(1).strip()
                    elif "Default Power Limit" in line:
                        match = re.match(
                            r"Default Power Limit\s*:\s*([\d.]+)\s*W", line
                        )
                        if match:
                            power_info["default_power_limit"] = match.group(1).strip()
                    elif "Min Power Limit" in line:
                        match = re.match(r"Min Power Limit\s*:\s*([\d.]+)\s*W", line)
                        if match:
                            power_info["min_power_limit"] = match.group(1).strip()
                    elif "Max Power Limit" in line:
                        match = re.match(r"Max Power Limit\s*:\s*([\d.]+)\s*W", line)
                        if match:
                            power_info["max_power_limit"] = match.group(1).strip()

            return power_info

        def parse_clocks_and_voltage(nvidia_smi_output):
            clocks_info = {
                "graphics_clock": None,
                "sm_clock": None,
                "memory_clock": None,
                "video_clock": None,
                "max_graphics_clock": None,
                "max_sm_clock": None,
                "max_memory_clock": None,
                "max_video_clock": None,
                "auto_boost": None,
                "graphics_voltage": None,
            }

            lines = nvidia_smi_output.splitlines()
            capturing_clocks = False
            capturing_max_clocks = False
            capturing_clock_policy = False
            capturing_voltage = False
            for line in lines:
                line = line.strip()

                # Start capturing Clocks
                if line.startswith("Clocks"):
                    capturing_clocks = True
                    capturing_max_clocks = False
                    capturing_clock_policy = False
                    capturing_voltage = False
                    continue

                # Start capturing Max Clocks
                if line.startswith("Max Clocks"):
                    capturing_max_clocks = True
                    capturing_clocks = False
                    capturing_clock_policy = False
                    capturing_voltage = False
                    continue

                # Start capturing Clock Policy
                if line.startswith("Clock Policy"):
                    capturing_clock_policy = True
                    capturing_clocks = False
                    capturing_max_clocks = False
                    capturing_voltage = False
                    continue

                # Start capturing Voltage
                if line.startswith("Voltage"):
                    capturing_voltage = True
                    capturing_clocks = False
                    capturing_max_clocks = False
                    capturing_clock_policy = False
                    continue

                if capturing_clocks:
                    if "Graphics" in line:
                        match = re.match(r"Graphics\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["graphics_clock"] = match.group(1).strip()
                    elif "SM" in line:
                        match = re.match(r"SM\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["sm_clock"] = match.group(1).strip()
                    elif "Memory" in line:
                        match = re.match(r"Memory\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["memory_clock"] = match.group(1).strip()
                    elif "Video" in line:
                        match = re.match(r"Video\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["video_clock"] = match.group(1).strip()

                elif capturing_max_clocks:
                    if "Graphics" in line:
                        match = re.match(r"Graphics\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["max_graphics_clock"] = match.group(1).strip()
                    elif "SM" in line:
                        match = re.match(r"SM\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["max_sm_clock"] = match.group(1).strip()
                    elif "Memory" in line:
                        match = re.match(r"Memory\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["max_memory_clock"] = match.group(1).strip()
                    elif "Video" in line:
                        match = re.match(r"Video\s*:\s*(\d+)\s*MHz", line)
                        if match:
                            clocks_info["max_video_clock"] = match.group(1).strip()

                elif capturing_clock_policy:
                    if "Auto Boost" in line:
                        match = re.match(r"Auto Boost\s*:\s*(N/A|\S+)", line)
                        if match:
                            clocks_info["auto_boost"] = match.group(1).strip()

                elif capturing_voltage:
                    if "Graphics" in line:
                        match = re.match(r"Graphics\s*:\s*([\d.]+)\s*mV", line)
                        if match:
                            clocks_info["graphics_voltage"] = match.group(1).strip()

            return clocks_info

        result = subprocess.run(
            ["nvidia-smi", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if result.returncode != 0:
            raise ValueError(result.stderr)

        gpu_specs = []
        devices = re.split(r"\nGPU [0-9A-Fa-f:.]+", result.stdout)

        for idx, device in enumerate(devices[1:]):
            gpu_specs.append(
                {
                    "device_id": idx,
                    "product_name": extract_value(r"Product Name\s+:\s+(.+)", device),
                    "product_brand": extract_value(r"Product Brand\s+:\s+(.+)", device),
                    "architecture": extract_value(
                        r"Product Architecture\s+:\s+(.+)", device
                    ),
                    "multi_gpu_board": extract_value(
                        r"MultiGPU Board \s+:\s+(.+)", device
                    ),
                    "virtualization_mode": extract_value(
                        r"Virtualization Mode\s+:\s+(.+)", device
                    ),
                    "host_vgpu_mode": extract_value(
                        r"Host VGPU Mode\s+:\s+(.+)", device
                    ),
                    "fan_speed": extract_value(r"Fan Speed\s+:\s+(.+)", device),
                    "performance_state": extract_value(
                        r"Performance State\s+:\s+(.+)", device
                    ),
                    **parse_pci_info(device),
                    **parse_clocks_event_reasons(device),
                    **parse_memory_usage(device),
                    **parse_temperature_and_power(device),
                    **parse_clocks_and_voltage(device),
                }
            )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        return gpu_specs, result.stdout


def get_gpu_info():
    gpu_info, raw_info = get_gpu_specs_smi()
    additional_specs = get_gpu_specs_pycuda()

    if len(gpu_info) == len(additional_specs):
        for i in range(len(gpu_info)):
            gpu_info[i].update(additional_specs[i])

    return gpu_info, raw_info
