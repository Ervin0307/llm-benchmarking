import json
import sys
import os

def main():
    from llm_benchmark.hardware import tools as hardware_tools
    return json.dumps(hardware_tools.get_hardware_info())


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

