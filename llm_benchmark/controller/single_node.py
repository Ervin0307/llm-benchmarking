import os
import time
import requests
import subprocess


def build_docker_run_command(
    model_name: str,
    docker_image: str,
    port: int,
    env_values: str,
    result_dir: str,
    extra_args: list,
) -> list:
    """Constructs the docker run command."""
    env_vars = [f"-e={env}" for env in env_values.split(",")] if env_values else []
    volumes = [
        f"-v {os.path.expanduser('~')}/.cache:/root/.cache",
        f"-v {result_dir}:/root/results",
    ]

    docker_command = [
        "docker",
        "run",
        "-d",
        "-it",
        "--rm",
        "--privileged",
        "--network=host",
        *env_vars,
        f"-e PROFILER_RESULT_DIR={result_dir}",
        *volumes,
        docker_image,
        "--model",
        model_name,
        "--port",
        str(port),
        *extra_args,
    ]

    return docker_command


def deploy_model(
    model_name: str,
    docker_image: str,
    port: int,
    env_values: str,
    result_dir: str,
    extra_args: list,
    warmup_sec: int = 60,
) -> str:
    try:
        docker_command = build_docker_run_command(
            model_name, docker_image, port, env_values, result_dir, extra_args
        )
        print(f"Deploying {model_name} with Docker image {docker_image}...")
        print("Executing Docker command: " + " ".join(docker_command))

        container = subprocess.run(
            docker_command, capture_output=True, text=True, check=True
        )
        container_id = container.stdout.strip()

        # Wait for the container to initialize
        time.sleep(warmup_sec)

        if not verify_server_status(f"http://localhost:{port}/v1"):
            raise RuntimeError("Server failed to start after maximum retries.")

        print(f"Container {container_id} for {model_name} is now running.")
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to deploy model {model_name}. Docker error: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error deploying model {model_name}: {e}")
        raise


def remove_container(container_id: str):
    try:
        subprocess.run(["docker", "rm", "-f", container_id], check=True)
        print(f"Container {container_id} removed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to remove container {container_id}. Docker error: {e.stderr}")
        raise


def verify_server_status(
    base_url: str, max_retries: int = 60, retry_interval: int = 30
) -> bool:
    """Verifies if the server is up and running by checking the API status."""
    url = f"{base_url}/models"

    for attempt in range(max_retries):
        try:
            response = requests.get(url)

            if response.status_code == 200:
                print("Server is up and running.")
                return True
            else:
                print(f"Server not ready. Status code: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error connecting to server: {e}")

        if attempt < max_retries - 1:
            print(
                f"Retrying in {retry_interval} seconds... (Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_interval)

    print(f"Server failed to start after {max_retries} retries.")
    return False
