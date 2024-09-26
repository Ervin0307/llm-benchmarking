import os
import uuid
import json
import csv

from .config import get_engine_config


def get_engine_dir(engine_config_id=None):
    if not engine_config_id:
        engine_config_id = os.environ.get("ENGINE_CONFIG_ID", str(uuid.uuid4())[:8])
    return os.path.join(
        os.environ.get("PROFILER_RESULT_DIR", "/tmp"),
        "engine",
        engine_config_id,
    )


def save_engine_config(args):
    config_dict = {}
    for item in vars(args):
        config_dict[item] = getattr(args, item)
    engine_dir = get_engine_dir()
    os.makedirs(engine_dir, exist_ok=True)
    print("Saving engine config to", engine_dir)
    with open(os.path.join(engine_dir, "engine_config.json"), "w") as f:
        json.dump(config_dict, f)

    print(f"Engine config saved to {os.path.join(engine_dir, 'engine_config.json')}")


def save_engine_envs(envs):
    envs_dict = {}
    for k, v in envs.items():
        envs_dict[k] = v()
    engine_dir = get_engine_dir()
    os.makedirs(engine_dir, exist_ok=True)
    with open(os.path.join(engine_dir, "engine_envs.json"), "w") as f:
        json.dump(envs_dict, f)

    print(f"Engine envs saved to {os.path.join(engine_dir, 'engine_envs.json')}")


def create_engine_summary(engine, engine_config_id, model):
    engine_dir = get_engine_dir(engine_config_id)
    with open(os.path.join(engine_dir, "engine_config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(engine_dir, "engine_envs.json"), "r") as f:
        envs = json.load(f)

    engine_config = get_engine_config(engine, config, envs)
    engine_config = {
        "engine_config_id": engine_config_id,
        **engine_config.get_config(),
    }

    results_dir = os.environ.get("PROFILER_RESULT_DIR", "/tmp")
    csv_file_path = os.path.join(
        results_dir,
        model.replace("/", "--"),
        f"engine_config.csv",
    )
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in write mode to ensure headers are always written
    with open(csv_file_path, "w", newline="") as csvfile:
        fieldnames = list(engine_config.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Always write headers
        writer.writeheader()

        # Write the summary data
        writer.writerow(engine_config)

    print(f"Engine summary saved to {csv_file_path}")

    return engine_config
