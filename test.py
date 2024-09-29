import subprocess
docker_command = ['docker', 'run', '-d', '-it', '--rm', '--privileged', '--network=host',  '-v=/home/ditto-bud/.cache:/root/.cache', '-v=/home/ditto-bud/results:/root/results', 'bud-runtime-cpu:latest', '--model=meta-llama/Meta-Llama-3-8B-Instruct']
container = subprocess.run(
            docker_command, capture_output=True, text=True, check=True
)
print(container)