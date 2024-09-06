import requests
import argparse
import time
import subprocess
import os

MAX_RETRIES = 60
RETRY_INTERVAL = 30
INITIAL_DELAY = 60

def run_benchmark(model, base_url, input_token, output_token, concurrency):
    script = "vllm"
    # Set environment variables directly
    os.environ["OPENAI_API_KEY"] = "secret_abcdefg"
    os.environ["OPENAI_API_BASE"] = base_url
    print("Running benchmark for model: ", model, "with input token: ", input_token, "and output token: ", output_token, "and concurrency: ", concurrency)
    print(" ".join([
            "python",
            "vllm_benchmark/benchmark_serving.py",
            "--backend", "vllm",
            "--model", model,
            "--dataset-name", "random",
            "--num-prompts", str(concurrency),
            "--random-input-len", str(input_token),
            "--random-output-len", str(output_token)
        ]))
    if script == "vllm":
        benchmark_result = subprocess.run([
            "python",
            "vllm_benchmark/benchmark_serving.py",
            "--backend", "vllm",
            "--model", model,
            "--dataset-name", "random",
            "--num-prompts", str(concurrency),
            "--random-input-len", str(input_token),
            "--random-output-len", str(output_token),
            "--save-result"
        ], capture_output=True, text=True, check=True)
    else:
        benchmark_result = subprocess.run([
            "python", 
            "llmperf/token_benchmark_ray.py", 
            "--model", model, 
            "--mean-input-tokens", str(input_token), 
            "--stddev-input-tokens", "0", 
            "--mean-output-tokens", str(output_token), 
            "--stddev-output-tokens", "0",
            "--max-num-completed-requests", str(concurrency),
            "--timeout", "600",
            "--num-concurrent-requests", str(concurrency),
            "--results-dir", "result_outputs",
            "--llm-api", "openai"], capture_output=True, text=True, check=True)
    result_output = benchmark_result.stdout
    print(result_output)
    return result_output

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
        container = subprocess.run([
            "docker", "run", 
            "-d", "-it", "--rm",
            "--privileged", "--network=host", 
            "-e", "HF_TOKEN=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", 
            "-v", "/home/ditto-bud/.cache:/root/.cache", 
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
        if input_token < 50:
            print("Skipping input token: ", input_token, " because it is less than 50")
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
    container_id = deploy_model(args.model, args.docker_image, args.port, args.extra_args)

    try:
        configs = create_config(args)
        for config in configs:
            print(config)
            run_benchmark(args.model, base_url, config["input_tokens"], config["output_tokens"], config["concurrency"])
    except Exception as e:
        print(f"Error during benchmark: {e}")
    finally:
        pass
        # remove_container(container_id)


if __name__ == "__main__":

    '''
    python benchmark/engine_benchmark.py --model <model> --docker-image <docker-image> --port <port> --input-tokens <input-tokens> --output-tokens <output-tokens> --concurrency <concurrency>
    '''
    args = argparse.ArgumentParser(
        description="Run a token throughput and latency benchmark."
    )

    args.add_argument(
        "--model", type=str, required=True, help="The model to use for this load test."
    )
    args.add_argument(
        "--docker-image", type=str, required=True, help="The engine image to be used for the testing."
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

    args = args.parse_args()

    main(args)