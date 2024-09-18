import re
import requests
import argparse
import time
import subprocess
import os, shutil
import csv
from tqdm import tqdm

from llm_benchmark.benchmark.vllm_benchmark.benchmark_serving import run_benchmark as vllm_run_benchmark
from llm_benchmark.benchmark.llmperf.token_benchmark_ray import run_token_benchmark as llmperf_run_benchmark
from llm_benchmark.profiler.utils.record_function_tracer import RecordFunctionTracer


MAX_RETRIES = 60
RETRY_INTERVAL = 30
INITIAL_DELAY = 60

PROFILER_PATH = '~/results'
RESULT_DIR = os.path.abspath(os.path.expanduser(f"{PROFILER_PATH}"))

def get_profiler_result():
    
    record_function_tracer = RecordFunctionTracer(RESULT_DIR, get_all=True)
    profile_stats = record_function_tracer.get_operation_time_stats()

    return profile_stats


def create_summary(results, results_dir):
    
    os.makedirs(results_dir, exist_ok=True)

    summary_list = []
    layers = [
        'embed', 
        'input_layernorm',
        'attn', 
        'attn_input_reshape',
        'attn_kv_cache_save',
        'attn_prefill',
        'attn_decode',
        'attn_output_reshape',
        'mlp',
        'post_layernorm', 
    ]

    for result in results:
        summary = {}
        summary['Model'] = result['model']
        summary['Mean Input Tokens'] = result['input_tokens']
        summary['Mean Output Tokens'] = result['output_tokens']
        summary['Concurrent Requests'] = result['concurrency']
        summary['Completed Requests'] = result['completed']
        summary['Duration (s)'] = round(result['duration'], 2)
        summary['Request Throughput (req/min)'] = round(result['request_throughput_per_min'], 2)
        summary['Output Token Throughput (tok/s)'] = round(result['output_throughput'], 2)
        summary['Output Token Throughput per User (tok/s)'] = round(result['output_throughput_per_user'], 2)
        summary['Mean End to End Latency (s)'] = round(result['mean_end_to_end_latency'], 2)
        summary['Mean TTFT (ms)'] = round(result['mean_ttft_ms'], 2)
        summary['P95 TTFT (ms)'] = round(result['p95_ttft_ms'], 2)
        summary['Mean Inter Token Latency (ms)'] = round(result['mean_itl_ms'], 2)
        summary['P95 Inter Token Latency (ms)'] = round(result['p95_itl_ms'], 2)

        for layer in layers:
            summary[f"{layer}_min"] = result[layer]['min'] if layer in result else ''
            summary[f"{layer}_max"] = result[layer]['max'] if layer in result else ''
            summary[f"{layer}_mean"] = result[layer]['mean'] if layer in result else ''
            summary[f"{layer}_median"] = result[layer]['median'] if layer in result else ''
            summary[f"{layer}_std"] = result[layer]['std'] if layer in result else ''

        summary_list.append(summary)

    if len(summary_list) == 0:
        print("No results to save")
        return

    # Define the CSV file path
    filename = f"{results[0]['model']}"
    filename = re.sub(r"[^\w\d-]+", "-", filename)
    filename = re.sub(r"-{2,}", "-", filename)
    
    csv_file_path = os.path.join(results_dir, f"{filename}_summary.csv")

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = list(summary_list[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the summary data
        for summary in summary_list:
            writer.writerow(summary)

    print(f"Benchmark summary saved to {csv_file_path}")
    return summary

def format_vllm_result(result):
    formatted_result = {}
    formatted_result['model'] = result['model_id']
    formatted_result['concurrency'] = result['concurrency']
    formatted_result['input_tokens'] = result['input_tokens']
    formatted_result['output_tokens'] = result['output_tokens']
    formatted_result['total_input_tokens'] = result['total_input_tokens']
    formatted_result['total_output_tokens'] = result['total_output_tokens']
    formatted_result['completed'] = result['completed']
    formatted_result['request_throughput'] = result['request_throughput']
    formatted_result['output_throughput'] = result['output_throughput']
    formatted_result['total_token_throughput'] = result['total_token_throughput']
    formatted_result['mean_request_throughput'] = result['mean_request_throughput']
    formatted_result['mean_ttft_ms'] = result['mean_ttft_ms']
    formatted_result['p95_ttft_ms'] = result['p95_ttft_ms']
    formatted_result['mean_tpot_ms'] = result['mean_tpot_ms']
    formatted_result['p95_tpot_ms'] = result['p95_tpot_ms']
    formatted_result['mean_itl_ms'] = result['mean_itl_ms']
    formatted_result['p95_itl_ms'] = result['p95_itl_ms']

    return formatted_result

def format_llmperf_result(result):
    formatted_result = {}
    formatted_result['model'] = result['model']
    formatted_result['concurrency'] = result['num_concurrent_requests']
    formatted_result['input_tokens'] = result['mean_input_tokens']
    formatted_result['output_tokens'] = result['mean_output_tokens']
    formatted_result['completed'] = result['results']['num_completed_requests']
    formatted_result['duration'] = result['results']['end_to_end_latency_s']['max']
    formatted_result['request_throughput_per_min'] = result['results']['num_completed_requests_per_min']
    formatted_result['output_throughput'] = result['results']['mean_output_throughput_token_per_s']
    formatted_result['output_throughput_per_user'] = result['results']['request_output_throughput_token_per_s']['mean']
    formatted_result['mean_end_to_end_latency'] = result['results']['end_to_end_latency_s']['mean']
    formatted_result['mean_ttft_ms'] = result['results']['ttft_s']['mean'] * 1000
    formatted_result['p95_ttft_ms'] = result['results']['ttft_s']['quantiles']['p95'] * 1000
    formatted_result['mean_itl_ms'] = result['results']['inter_token_latency_s']['mean'] * 1000
    formatted_result['p95_itl_ms'] = result['results']['inter_token_latency_s']['quantiles']['p95'] * 1000
    return formatted_result


def run_benchmark(model, base_url, input_token, output_token, concurrency):
    script = args.benchmark_script
    # Set environment variables directly
    os.environ["OPENAI_API_KEY"] = "secret_abcdefg"
    os.environ["OPENAI_API_BASE"] = base_url

    traces_dir = f"{RESULT_DIR}/profiler_traces/"
    if os.path.exists(traces_dir):
        shutil.rmtree(traces_dir)
    os.makedirs(traces_dir, exist_ok=True)
    

    print("Running benchmark for model: ", model, "with input token: ", input_token, "and output token: ", output_token, "and concurrency: ", concurrency)

    if script == "vllm":
        result_output = vllm_run_benchmark(model, input_token, output_token, concurrency, base_url)
        result_output = format_vllm_result(result_output)
    else:
        result_output = llmperf_run_benchmark(model, concurrency, concurrency, input_token, 0, output_token, 0)
        result_output = format_llmperf_result(result_output)
    
    profiler_stats = get_profiler_result()

    return {
        **result_output,
        **profiler_stats
    }

def verify_server_status(base_url):
    '''function to validate if the server is up and running by checking the api status'''
    url = f"{base_url}/models"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                print("Server is up and running.")
                return True
            else:
                print(f"Server not ready. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error connecting to server: {e}")
        
        if attempt < MAX_RETRIES - 1:
            print(f"Retrying in {RETRY_INTERVAL} seconds...")
            time.sleep(RETRY_INTERVAL)
    
    print("Server failed to start after maximum retries.")
    return False
    
def deploy_model(model_name, docker_image, port, extra_args):
    try:
        # Step 1: Deploy the model using Docker
        print(f"Deploying {model_name} with image {docker_image}...")
        print(" ".join([
            "docker", "run", 
            "-d", "-it", "--rm",
            "--privileged", "--network=host", 
            *[f"-e={env_value}" for env_value in args.env_values.split(',')],
            "-v", f"{os.path.expanduser('~')}/.cache:/root/.cache",
            docker_image, 
            "--model", model_name,
            "--port", port,
            *extra_args.split()
        ]))
        container = subprocess.run([
            "docker", "run", 
            "-d", "-it", "--rm",
            "--privileged", "--network=host", 
            *([f"-e={env_value}" for env_value in args.env_values.split(',')] if args.env_values else []),
            "-e", f"PROFILER_RESULT={RESULT_DIR}",
            "-v", f"{os.path.expanduser('~')}/.cache:/root/.cache",
            "-v", f"{RESULT_DIR}:/root/results",
            docker_image, 
            "--model", model_name,
            "--port", port,
            *extra_args.split()
        ], capture_output=True, text=True, check=True)
        container_id = container.stdout.strip()
        
        time.sleep(INITIAL_DELAY)

        if not verify_server_status(f"http://localhost:{port}/v1"):
            raise Exception("Server failed to start after maximum retries.")

        print(f"Container for {model_name} is now running.")
        return container_id
    except Exception as e:
        print(f"Failed to deploy model {model_name}: {e}")
        raise

def remove_container(container_id):
    try:
        subprocess.run(["docker", "rm", "-f", container_id], check=True)
        print(f"Container {container_id} removed.")
    except Exception as e:
        print(f"Failed to remove container {container_id}: {e}")
        raise

def create_config(args):
    configs = []
    input_tokens = [int(x) for x in args.input_tokens.split(',')]
    output_tokens = [int(x) for x in args.output_tokens.split(',')]
    concurrencies = [int(x) for x in args.concurrency.split(',')]

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
        container_id = deploy_model(args.model, args.docker_image, args.port, args.extra_args)
    else:
        container_id = None

    os.makedirs(RESULT_DIR, exist_ok=True)

    results = []
    try:
        configs = create_config(args)
        for config in tqdm(configs, desc="Running benchmarks"):
            print(config)
            result = run_benchmark(args.model, base_url, config["input_tokens"], config["output_tokens"], config["concurrency"])
            result['input_tokens'] = config['input_tokens']
            result['output_tokens'] = config['output_tokens']
            result['concurrency'] = config['concurrency']
            results.append(result)
            print(result)
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        if container_id:
            remove_container(container_id)
    
    create_summary(results, RESULT_DIR)

if __name__ == "__main__":

    '''
    python benchmark/auto_benchmark.py --model <model> --docker-image <docker-image> --port <port> --input-tokens <input-tokens> --output-tokens <output-tokens> --concurrency <concurrency>
    '''
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    args.add_argument(
        "--docker-image", type=str, default=None, help="The engine image to be used for the testing."
    )
    args.add_argument(
        "--port", type=str, default="8000", help="The port where the engine will be running"
    )
    args.add_argument(
        "--input-tokens", type=str, default="128,256,512,1024", help="List of different input token combinations"
    )
    args.add_argument(
        "--output-tokens", type=str, default="128,256,512,1024", help="List of different output token combinations"
    )
    args.add_argument(
        "--concurrency", type=str, default="1,10,20,30,50,100", help="List of concurrency for the benchmark"
    )
    args.add_argument(
        "--extra-args", type=str, default="", help="Extra arguments to be passed to the engine"
    )
    args.add_argument(
        "--env-values", type=str, default="", help="Environment values to be set for the benchmark" 
    )
    args.add_argument(
        "--benchmark-script", type=str, default="llmperf", help="The benchmark script to be used for the testing."
    )


    args = args.parse_args()

    main(args)