import os
import argparse
import yaml
import threading
import itertools

from tqdm import tqdm
import uuid

from llm_benchmark.controller import single_node as single_node_controller
from llm_benchmark.benchmark import tools as benchmark_tools
from llm_benchmark.profiler import tools as profiler_tools
from llm_benchmark.hardware import tools as hardware_tools
from llm_benchmark.hardware import monitor as hw_monitor
from llm_benchmark.engine import tools as engine_tools


def create_config(run_config):
    configs = []
    input_tokens = [int(x) for x in run_config['mean_input_tokens']] if isinstance(run_config['mean_input_tokens'], list) else [run_config['mean_input_tokens']]
    output_tokens = [int(x) for x in run_config['mean_output_tokens']] if isinstance(run_config['mean_output_tokens'], list) else [run_config['mean_output_tokens']]
    concurrencies = [int(x) for x in run_config['num_concurrent_requests']] if isinstance(run_config['num_concurrent_requests'], list) else [run_config['num_concurrent_requests']]

    for input_token in input_tokens:
        if input_token < 20:
            print("Skipping input token: ", input_token, " because it is less than 20")
            continue
        for output_token in output_tokens:
            for concurrency in concurrencies:
                config = {
                    "input_tokens": input_token,
                    "output_tokens": output_token,
                    "concurrency": concurrency,
                }
                configs.append(config)
    return configs

def warmup_benchmark(model, base_url, benchmark_script):
    print("Running warmup benchmark")
    result = benchmark_tools.run_benchmark(
        model,
        base_url,
        250,
        250,
        10,
        benchmark_script,
        os.environ["PROFILER_RESULT_DIR"],
        "warmup",
    )

# Function to process and create combinations
def generate_combinations(config_section):
    fixed_params = {}
    array_params = {}

    for key, value in config_section.items():
        if isinstance(value, list):
            array_params[key] = value
        else:
            fixed_params[key] = value

    # Generate all possible combinations for array parameters
    if array_params:
        keys, values = zip(*array_params.items())
        combinations = list(itertools.product(*values))
    else:
        combinations = [()]  # No combinations to generate

    return fixed_params, array_params, combinations, keys if array_params else []


def create_engine_config(engine_config_file):
    with open(engine_config_file, "r") as f:
        engine_config = yaml.safe_load(f)

    # Separate the fixed parameters and the parameters with arrays
    # Process the 'args' section
    fixed_args, array_args, arg_combinations, arg_keys = generate_combinations(
        engine_config["args"]
    )

    # Process the 'envs' section
    fixed_envs, array_envs, env_combinations, env_keys = generate_combinations(
        engine_config["envs"]
    )

    # Create a list of configuration dictionaries with all combinations
    configs = []
    for arg_comb in arg_combinations:
        for env_comb in env_combinations:
            # Create new config dict for each combination
            new_config = {
                "args": fixed_args.copy(),  # Copy fixed args
                "envs": fixed_envs.copy(),  # Copy fixed envs
            }
            # Update with current combination of 'args'
            if arg_comb:
                new_config["args"].update(dict(zip(arg_keys, arg_comb)))

            # Update with current combination of 'envs'
            if env_comb:
                new_config["envs"].update(dict(zip(env_keys, env_comb)))

            # Append the complete config to the list
            configs.append(new_config)
    
    return configs, engine_config['run_config']


def run_benchmark(args, engine_config, run_config):
    base_url = f"http://localhost:{args.port}/v1"
    model = engine_config["args"]['model']

    if args.engine_config_id:
        engine_config_id = args.engine_config_id
    else:
        engine_config_id = str(uuid.uuid4())[:8]

    
    if args.docker_image:
        container_id = single_node_controller.deploy_model(
            args.docker_image,
            engine_config["envs"] if engine_config else [],
            os.environ["PROFILER_RESULT_DIR"],
            engine_config["args"] if engine_config else [],
            engine_config_id,
            args.port,
            cpu_only=args.cpu_only,
            profile_model=args.profile_model,
        )
    else:
        container_id = None

    if args.engine_config_id or container_id:
        engine_tools.create_engine_summary(args.engine, engine_config_id, model)
    
    warmup_benchmark(model, base_url, args.benchmark_script)

    log_metrics_task = None
    stop_event = None
    results = []
    try:
        configs = create_config(run_config)
        for config in tqdm(configs, desc="Running benchmarks"):
            print(config)
            run_id = str(uuid.uuid4())[:8]

            stop_event = threading.Event()
            log_metrics_task = threading.Thread(
                target=hw_monitor.log_system_metrics,
                kwargs={
                    "output_dir": os.path.join(
                        os.environ["PROFILER_RESULT_DIR"], model.replace("/", "--")
                    ),
                    "pid": single_node_controller.get_container_pid(container_id)
                    if container_id is not None
                    else None,
                    "interval": 3,
                    "stop_event": stop_event,
                    "metadata": {
                        "run_id": run_id,
                        "engine_config_id": engine_config_id,
                    },
                },
            )
            log_metrics_task.start()

            result = benchmark_tools.run_benchmark(
                model,
                base_url,
                config["input_tokens"],
                config["output_tokens"],
                config["concurrency"],
                args.benchmark_script,
                os.environ["PROFILER_RESULT_DIR"],
                run_id,
            )

            result["engine"] = args.engine
            result["engine_config_id"] = engine_config_id
            result["run_id"] = run_id
            result["input_tokens"] = config["input_tokens"]
            result["output_tokens"] = config["output_tokens"]
            result["concurrency"] = config["concurrency"]

            results.append(result)

            stop_event.set()
            log_metrics_task.join()
            log_metrics_task = None
            stop_event = None

            benchmark_tools.create_summary([result], os.environ["PROFILER_RESULT_DIR"])
            print(result)
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        if container_id:
            single_node_controller.remove_container(container_id)
        if log_metrics_task is not None and stop_event is not None:
            stop_event.set()
            log_metrics_task.join()


def main(args):
    os.makedirs(os.environ["PROFILER_RESULT_DIR"], exist_ok=True)

    if args.run_benchmark:
        if args.engine_config_file:
            engine_configs, run_config = create_engine_config(args.engine_config_file)
        else:
            raise ValueError("Engine config file is required")
    
        for engine_config in tqdm(engine_configs, desc="Running engine configs"):
            run_benchmark(args, engine_config, run_config)
            # break

    if args.profile_collectives:
        profiler_tools.profile_collectives(
            max_collective_size=4096 * 8192,
            output_dir=os.environ["PROFILER_RESULT_DIR"],
        )

    if args.profile_hardware:
        hardware_tools.get_hardware_info(output_dir=os.environ["PROFILER_RESULT_DIR"])


if __name__ == "__main__":
    """
    python benchmark/auto_benchmark.py --model <model> --docker-image <docker-image> --port <port> --input-tokens <input-tokens> --output-tokens <output-tokens> --concurrency <concurrency>
    """
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="The engine image to be used for the testing.",
    )
    args.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="The engine to be used for the testing.",
    )
    args.add_argument(
        "--engine-config-file",
        type=str,
        default=None,
        help="The engine config file to be used for the testing.",
    )
    args.add_argument(
        "--engine-config-id",
        type=str,
        default=None,
        help="The engine config id to be used for the testing.",
    )
    args.add_argument(
        "--port",
        type=str,
        default="8000",
        help="The port where the engine will be running",
    )
    args.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Whether to run the benchmark.",
    )
    args.add_argument(
        "--benchmark-script",
        type=str,
        default="llmperf",
        help="The benchmark script to be used for the testing.",
    )

    args.add_argument(
        "--profile-collectives",
        action="store_true",
        help="Whether to profile the collectives.",
    )
    args.add_argument(
        "--cpu-only",
        action="store_true",
        help="Whether to profile only on cpu.",
    )
    args.add_argument(
        "--profile-hardware",
        action="store_true",
        help="Whether to profile the hardware.",
    )
    args.add_argument(
        "--profile-model",
        action="store_true",
        help="Whether to profile the model.",
    )
    args = args.parse_args()

    main(args)
