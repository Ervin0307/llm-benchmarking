import os
import torch.nn as nn

from typing import Optional
from functools import wraps

import ray
import pandas as pd
import datetime
from tqdm import tqdm

from .constants import VllmProfileLayer
from .record_function_tracer import RecordFunctionTracer
from .device_timer import DeviceTimer
from .timer_stats_store import TimerStatsStore
from .collectives.collectives_input import get_collectives_inputs
from .collectives.benchmark_runner import create_runner_pool
from llm_benchmark.utils.device_utils import (
    get_available_devices,
    get_tensor_parallel_sizes,
)


original_forwards = {}
timer_store_map = {}


def create_wrapped_forward(submodule: nn.Module, timer_store: DeviceTimer):
    """
    Factory function to create a wrapped forward method for each submodule.

    Args:
        submodule: The submodule (layer) whose forward method is being wrapped.
        key: The unique key for this submodule used in the timer store map.
        timer_store_map: The dictionary holding timers for each layer.
    """
    original_forward = submodule.forward

    @wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        with timer_store:
            return original_forward(*args, **kwargs)

    return wrapped_forward


def profile_model(model: nn.Module, model_id: str, cpu_only: bool = False):
    """
    Apply RecordFunctionTracer to the entire model class.

    Args:
        model (nn.Module): The PyTorch model to profile.
        tracer_name (str): Name for the RecordFunctionTracer.
    """

    TimerStatsStore(profile_method="record_function")

    key = type(model).__name__

    if model not in original_forwards:
        original_forwards[model] = model.forward

        def wrapped_forward(*args, **kwargs):
            with RecordFunctionTracer(
                os.path.join(
                    os.environ.get("PROFILER_RESULT_DIR", "/tmp"),
                    model_id.replace("/", "--"),
                ),
                cpu_only=cpu_only,
            ):
                return original_forwards[model](*args, **kwargs)

        model.forward = wrapped_forward

    # Apply layer-wise profiling with DeviceTimer
    profile_layer(model, parent_name=key, timer_store_map=timer_store_map)

    return model


def profile_layer(
    module: nn.Module, parent_name: str = None, timer_store_map: Optional[dict] = None
):
    """
    Recursively apply DeviceTimer to each layer and submodule in the model.

    Args:
        module (nn.Module): The module or layer to profile.
    """
    if timer_store_map is None:
        timer_store_map = {}

    for name, submodule in module.named_children():
        key = f"{parent_name or ''}.{name}".lstrip(".") + f":{type(submodule).__name__}"
        # If the submodule has no children, it is a "leaf" and should be wrapped

        profile_name = VllmProfileLayer.get_profile_name_by_operation(key)

        if not (profile_name is None or key in timer_store_map):
            timer_store_map[key] = DeviceTimer(profile_name)
            original_forwards[submodule] = submodule.forward
            print(key)

            submodule.forward = create_wrapped_forward(submodule, timer_store_map[key])

        if len(list(submodule.children())) != 0:
            profile_layer(
                submodule,
                parent_name=key.split(":")[0],
                timer_store_map=timer_store_map,
            )


def stop_profile():
    for module, original_forward in original_forwards.items():
        # Restore original forward methods for all submodules
        module.forward = original_forward

    # # Clear the dictionary after restoring the model
    original_forwards.clear()


def profile_collectives(
    max_collective_size: int,
    output_dir: str = "results",
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, "collective")
    os.makedirs(output_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    devices = get_available_devices()
    for device in devices:
        for collective in ("all_reduce", "send_recv"):
            total_workers_available, num_nodes, runner_pool = create_runner_pool(device)
            num_workers_per_node_combinations = get_tensor_parallel_sizes(
                total_workers_available
            )

            all_results = []
            collectives_inputs = get_collectives_inputs(
                num_nodes,
                num_workers_per_node_combinations,
                max_collective_size,
                collective,
                total_workers_available,
            )

            for collectives_input in tqdm(collectives_inputs):
                promises = []
                for worker_id in range(total_workers_available):
                    promise = runner_pool[worker_id].run_collective.remote(
                        collectives_input
                    )
                    promises.append(promise)

                for worker_id in range(int(total_workers_available)):
                    result = ray.get([promises[worker_id]])[0]
                    if result and worker_id == 0:
                        all_results.append(result)

                ray.get(promises)

            # filter none results
            all_results = [x for x in all_results if x is not None]

            df = pd.DataFrame(all_results)
            # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
            df = (
                pd.json_normalize(df["time_stats"])
                .add_prefix("time_stats.")
                .join(df.drop(columns=["time_stats"]))
            )

            # write results to a csv file
            df.to_csv(
                os.path.join(output_dir, f"{device}_{collective}_{timestamp}.csv")
            )
