import os
import uuid
import json

def get_engine_dir():
    return os.path.join(
        os.environ.get("PROFILER_RESULT_DIR", "/tmp"), 
        "engine",
        os.environ.get("ENGINE_CONFIG_ID", str(uuid.uuid4())[:8]),
    )

def save_engine_config(args):
    config_dict = {}
    for item in vars(args):
        config_dict[item] = getattr(args, item)
    engine_dir = get_engine_dir()
    os.makedirs(engine_dir, exist_ok=True)
    with open(os.path.join(engine_dir, 'engine_config.json'), 'w') as f:
        json.dump(config_dict, f)


def save_engine_envs(envs):
    envs_dict = {}
    for k, v in envs.items():
        envs_dict[k] = v()
    engine_dir = get_engine_dir()
    os.makedirs(engine_dir, exist_ok=True)
    with open(os.path.join(engine_dir, 'engine_envs.json'), 'w') as f:
        json.dump(envs_dict, f)