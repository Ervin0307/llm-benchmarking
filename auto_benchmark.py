import os
import argparse

from tqdm import tqdm
import uuid

from llm_benchmark.controller import single_node as single_node_controller
from llm_benchmark.benchmark import tools as benchmark_tools
from llm_benchmark.profiler import tools as profiler_tools
from llm_benchmark.hardware import tools as hardware_tools


def create_config(args):
    configs = []
    input_tokens = [int(x) for x in args.input_tokens.split(",")]
    output_tokens = [int(x) for x in args.output_tokens.split(",")]
    concurrencies = [int(x) for x in args.concurrency.split(",")]

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


def main(args):
    base_url = f"http://localhost:{args.port}/v1"
    if args.docker_image:
        container_id = single_node_controller.deploy_model(
            args.model,
            args.docker_image,
            args.port,
            args.env_values,
            os.environ["PROFILER_RESULT_DIR"],
            args.extra_args.split(),
        )
    else:
        container_id = None

    os.makedirs(os.environ["PROFILER_RESULT_DIR"], exist_ok=True)

    results = []
    try:
        configs = create_config(args)
        for config in tqdm(configs, desc="Running benchmarks"):
            print(config)
            run_id = str(uuid.uuid4())[:8]
            result = benchmark_tools.run_benchmark(
                args.model,
                base_url,
                config["input_tokens"],
                config["output_tokens"],
                config["concurrency"],
                args.benchmark_script,
                os.environ["PROFILER_RESULT_DIR"],
                run_id,
            )
            
            result["run_id"] = run_id
            result["input_tokens"] = config["input_tokens"]
            result["output_tokens"] = config["output_tokens"]
            result["concurrency"] = config["concurrency"]
            
            results.append(result)
            print(result)
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        if container_id:
            single_node_controller.remove_container(container_id)
    
    benchmark_tools.create_summary(results, os.environ["PROFILER_RESULT_DIR"])

    if args.profile_collectives:
        profiler_tools.profile_collectives(
            num_workers_per_node_combinations=[1, 2],
            max_collective_size=512 * 1024,
            collective="all_reduce", # "all_reduce" or "send_recv"
            device="cpu" if args.cpu_only else "cuda", # "cpu" or "cuda" 
            output_dir=os.environ["PROFILER_RESULT_DIR"]
        )
    
    if args.profile_hardware:
        hardware_tools.get_hardware_info(
            cpu_only=args.cpu_only,
            output_dir=os.environ["PROFILER_RESULT_DIR"]
        )


if __name__ == "__main__":
    """
    python benchmark/auto_benchmark.py --model <model> --docker-image <docker-image> --port <port> --input-tokens <input-tokens> --output-tokens <output-tokens> --concurrency <concurrency>
    """
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    args.add_argument(
        "--docker-image",
        type=str,
        default=None,
        help="The engine image to be used for the testing.",
    )
    args.add_argument(
        "--port",
        type=str,
        default="8000",
        help="The port where the engine will be running",
    )
    args.add_argument(
        "--input-tokens",
        type=str,
        default="128,256,512,1024",
        help="List of different input token combinations",
    )
    args.add_argument(
        "--output-tokens",
        type=str,
        default="128,256,512,1024",
        help="List of different output token combinations",
    )
    args.add_argument(
        "--concurrency",
        type=str,
        default="1,10,20,30,50,100",
        help="List of concurrency for the benchmark",
    )
    args.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra arguments to be passed to the engine",
    )
    args.add_argument(
        "--env-values",
        type=str,
        default="",
        help="Environment values to be set for the benchmark",
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
    args = args.parse_args()

    main(args)
