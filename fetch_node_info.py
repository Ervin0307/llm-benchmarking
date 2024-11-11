import json
import sys
import os



def main():
    from llm_benchmark.utils.device_utils import get_available_devices
    from llm_benchmark.hardware import tools as hardware_tools
    device_info = []
    devices = get_available_devices()
    for device in devices:
        device_config, dev_info = hardware_tools.create_device_config(device)
        device_config["type"] = device
        device_info.append({
            "name": device_config['name'],
            "type": device,
            "device_config": device_config,
            "device_info": dev_info,
            "available_count": device_config["available_count"]
        })
    return json.dumps(device_info)


if __name__ == "__main__":
    try:
        # Redirect unwanted stdout messages to stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        hardware = main()
        # Only print the JSON output
        sys.stdout = sys.__stdout__
        print(hardware)
    except Exception as e:
        # Log the error to stderr if necessary
        print(f"Error: {e}", file=sys.stderr)

