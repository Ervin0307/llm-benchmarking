import glob
import json
import uuid
import math

import numpy as np
import torch
from torch.autograd.profiler_util import (
    EventList,
    _format_time_share,
    _format_memory,
    _format_time,
)
from pathlib import Path

import torch.autograd.profiler_util


class RecordFunctionTracer:
    __slots__ = (
        "output_path",
        "trace_path",
        "profiler",
        "cpu_only",
        "profile_memory",
        "with_flops",
    )

    def __init__(
        self,
        output_path: str,
        get_all: bool = False,
        cpu_only: bool = False,
        profile_memory: bool = False,
        with_flops: bool = False,
    ):
        trace_id = str(uuid.uuid4())[:8] if not get_all else "*"
        self.output_path = output_path
        self.trace_path = (
            f"{output_path}/profiler_traces/profiler_trace_{trace_id}.json"
        )

        self.cpu_only = cpu_only
        self.profile_memory = profile_memory
        self.with_flops = with_flops

    def __enter__(self):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if not self.cpu_only:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        schedule = torch.profiler.schedule(
            wait=1,  # warmup iterations
            warmup=1,  # profiling iterations
            active=2,  # iterations to capture
            repeat=2  # repeat the schedule
        )
        self.profiler = torch.profiler.profile(
            activities=activities,
            # schedule=schedule,
            record_shapes=self.profile_memory,
            profile_memory=self.profile_memory,
            with_stack=self.profile_memory,
            with_flops=self.with_flops,
        )
        self.profiler.__enter__()

    def __exit__(self, *args):
        self.profiler.__exit__(None, None, None)

        if not self.cpu_only:
            torch.cuda.synchronize()

        self.export_profiler_trace(
            sort_by="self_cuda_time_total",
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
        )

    def export_profiler_trace(
        self,
        sort_by: str = None,
        row_limit: int = None,
        with_flops=False,
        profile_memory=False,
        top_level_events_only=False,
    ):
        Path(self.trace_path).parent.mkdir(exist_ok=True, parents=True)

        self.profiler.export_chrome_trace(self.trace_path)

        if profile_memory:
            self.profiler.export_memory_timeline(
                self.trace_path.replace(".json", ".html")
            )
        elif not with_flops:
            return

        events = self.profiler.events()
        if len(events) == 0:
            return

        has_device_time = any(event.self_device_time_total > 0 for event in events)
        has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
        use_device = events[0].use_device
        # Running on PrivateUse1 device with profiler but not enable
        # ProfilerActivity.PrivateUse1 can also catch privateuse1 memory usage.
        # Here only need to check has_privateuse1_time if not use_device.
        if not use_device and has_device_time:
            raise RuntimeError(
                "use_device is None, but there is device performance data."
            )

        has_input_shapes = any(
            (event.input_shapes is not None and len(event.input_shapes) > 0)
            for event in events
        )

        if sort_by is not None:
            events = EventList(
                sorted(
                    events,
                    key=lambda evt: getattr(
                        evt,
                        sort_by.replace("cuda", "device")
                        .replace("xpu", "device")
                        .replace("privateuse1", "device"),
                    ),
                    reverse=True,
                ),
                use_device=use_device,
                profile_memory=profile_memory,
                with_flops=with_flops,
            )

        stacks = []
        for evt in events:
            if evt.stack is not None and len(evt.stack) > 0:
                stacks.append(evt.stack)
        has_stack = len(stacks) > 0

        device_name = use_device.lower() if use_device is not None else "None"

        # Only append Node ID if any event has a valid (>= 0) Node ID
        append_node_id = any(evt.node_id != -1 for evt in events)

        def auto_scale_flops(flops):
            flop_headers = [
                "flops",
                "kflops",
                "mflops",
                "gflops",
                "tflops",
                "pflops",
            ]
            assert flops > 0
            log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
            assert log_flops >= 0 and log_flops < len(flop_headers)
            return (
                pow(10, (math.floor(log_flops) * -3.0)),
                flop_headers[int(log_flops)],
            )

        if with_flops:
            # Auto-scaling of flops header
            raw_flops = []
            for evt in events:
                if evt.flops is not None and evt.flops > 0:
                    raw_flops.append(evt.flops)
            if len(raw_flops) != 0:
                (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            else:
                with_flops = False  # can't find any valid flops

        # Have to use a list because nonlocal is Py3 only...
        result = {"traceEvents": []}

        sum_self_cpu_time_total = 0
        sum_self_device_time_total = 0
        for evt in events:
            sum_self_cpu_time_total += evt.self_cpu_time_total
            if evt.device_type == torch.autograd.DeviceType.CPU and evt.is_legacy:
                # in legacy profiler, kernel info is stored in cpu events
                sum_self_device_time_total += evt.self_device_time_total
            elif evt.device_type in [
                torch.autograd.DeviceType.CUDA,
                torch.autograd.DeviceType.PrivateUse1,
            ]:
                # in kineto profiler, there're events with the correct device type (e.g. CUDA)
                sum_self_device_time_total += evt.self_device_time_total

        event_limit = 0
        for evt in events:
            if row_limit is not None and event_limit == row_limit:
                break
            if top_level_events_only and evt.cpu_parent is not None:
                continue
            else:
                event_limit += 1
            name = evt.key

            row_values = {
                "name": name,
                # Self CPU total %, 0 for async events.
                "self_cpu_pct": _format_time_share(
                    evt.self_cpu_time_total, sum_self_cpu_time_total
                ),
                "self_cpu": evt.self_cpu_time_total_str,  # Self CPU total
                # CPU total %, 0 for async events.
                "cpu_total_pct": _format_time_share(
                    evt.cpu_time_total, sum_self_cpu_time_total
                )
                if not evt.is_async
                else 0,
                "cpu_total": evt.cpu_time_total_str,  # CPU total
                "cpu_time_avg": evt.cpu_time_str,  # CPU time avg
            }
            if has_device_time:
                row_values.update(
                    {
                        f"self_{device_name}": evt.self_device_time_total_str,
                        # CUDA time total %
                        f"self_{device_name}_pct": _format_time_share(
                            evt.self_device_time_total, sum_self_device_time_total
                        ),
                        f"{device_name}_total": evt.device_time_total_str,
                        f"{device_name}_time_avg": evt.device_time_str,  # Cuda time avg
                    }
                )

            if profile_memory:
                row_values.update(
                    {
                        # CPU Mem Total
                        "cpu_mem": _format_memory(evt.cpu_memory_usage),
                        # Self CPU Mem Total
                        "self_cpu_mem": _format_memory(evt.self_cpu_memory_usage),
                    }
                )
                if use_device and has_device_mem:
                    row_values.update(
                        {
                            # CUDA Mem Total
                            f"{device_name}_mem": _format_memory(
                                evt.device_memory_usage
                            ),
                            # Self CUDA Mem Total
                            f"self_{device_name}_mem": _format_memory(
                                evt.device_memory_usage
                            ),
                        }
                    )

            row_values["num_calls"] = (evt.count,)  # Number of calls

            if append_node_id:
                row_values["node_id"] = evt.node_id
            if has_input_shapes:
                row_values["input_shapes"] = str(evt.input_shapes)
            if with_flops and evt.flops is not None:
                row_values["total_flops"] = f"{evt.flops * flops_scale:8.3f}"  # type: ignore[possibly-undefined]
                row_values["flops_scale"] = flops_header
            if has_stack:
                src_field = ""
                if len(evt.stack) > 0:
                    src_field = evt.stack[0]
                row_values["source_loc"] = [src_field]

            if has_stack:
                for entry in evt.stack[1:]:
                    row_values["source_loc"].append(entry)

            result["traceEvents"].append(row_values)

        result["self_cpu_time_total"] = _format_time(sum_self_cpu_time_total)
        if has_device_time:
            result[f"self_{device_name}_time_total"] = _format_time(
                sum_self_device_time_total
            )

        with open(self.trace_path.replace(".json", ".trace.json"), "w") as fp:
            json.dump(result, fp)

    def find_children(self, trace, event):
        if not ("dur" in event and "ts" in event):
            return

        children = []
        for e in trace:
            if not ("dur" in e and "ts" in e):
                continue

            # if the ts of the child is completely within the ts of the parent
            if (
                e["ts"] > event["ts"]
                and e["ts"] + e["dur"] < event["ts"] + event["dur"]
            ):
                children.append(e)
        return children

    def find_correlated_event(self, trace, event):
        if not ("args" in event and "correlation" in event["args"]):
            return

        for e in trace:
            if not ("args" in e and "correlation" in e["args"]):
                continue

            if e == event:
                continue

            if e["args"]["correlation"] == event["args"]["correlation"]:
                return e

    def find_related_event(self, trace, event):
        if not ("args" in event and "External id" in event["args"]):
            return

        for e in trace:
            if not ("args" in e and "External id" in e["args"]):
                continue

            if e["args"]["External id"] == event["args"]["External id"]:
                return e

    def get_operation_time_stats(self):
        stats = {}

        traces = []
        for trace_file in glob.glob(self.trace_path):
            with open(trace_file, "r") as f:
                traces.append(json.load(f)["traceEvents"])

        if not len(traces):
            return {}

        user_events = {}
        for trace in traces:
            for event in trace:
                
                if not ("cat" in event and event["cat"] == "user_annotation"):
                    continue
                
                if not ("args" in event and "External id" in event["args"]):
                    continue
                
                ext_id = event["args"]["External id"]
                if ext_id not in user_events:
                    user_events[ext_id] = event
                else:
                    if event["dur"] > user_events[ext_id]["dur"]:
                        user_events[ext_id] = event

        for event in user_events.values():
            cuda_time = event["dur"]
            name = event["name"].replace("profiler.", "")
            
            if name not in stats:
                stats[name] = []

            stats[name].append(cuda_time * 1e-3)  # to convert to ms

        return {
            operation: {
                "min": np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
            for operation, times in stats.items()
        }
