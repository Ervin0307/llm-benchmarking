import os
import torch.nn as nn

from typing import Optional
from functools import wraps

from .constants import VllmProfileLayer
from .record_function_tracer import RecordFunctionTracer
from .device_timer import DeviceTimer
from .utils.timer_stats_store import TimerStatsStore


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
        key = f"{parent_name or ""}.{name}".lstrip(".") + f":{type(submodule).__name__}"
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
