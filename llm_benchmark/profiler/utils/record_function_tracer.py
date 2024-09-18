import glob
import json
import uuid

import numpy as np
import torch


class RecordFunctionTracer:
    def __init__(self, output_path: str, get_all: bool = False):
        trace_id = str(uuid.uuid4())[:8] if not get_all else '*'
        self.output_path = output_path
        self.trace_path = (
            f"{output_path}/profiler_traces/profiler_trace_{trace_id}.json"
        )

    def __enter__(self):
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                # torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    def __exit__(self, *args):
        self.profiler.__exit__(None, None, None)
        # torch.cuda.synchronize()
        self.profiler.export_chrome_trace(self.trace_path)

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
    
    def fine_related_event(self, trace, event):
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
        trace = traces[0]

        for event in trace:
            if not ("cat" in event and event["cat"] == "user_annotation"):
                continue
            children = self.find_children(trace, event)
            cuda_time = event['dur']
            # for child in children:
            #     if not ("cat" in child and child["cat"] == "cuda_runtime"):
            #         continue
            #     correlated_event = self.find_correlated_event(trace, child)
            #     if not correlated_event:
            #         continue
            #     cuda_time += correlated_event["dur"]
            for rank_trace in traces[1:]:
                related_event = self.fine_related_event(rank_trace, event)
                if related_event:
                    cuda_time += related_event['dur']
            if cuda_time == 0:
                continue

            name = event["name"].replace("llm_", "")

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