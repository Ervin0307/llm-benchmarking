import time

import torch
from torch.profiler import record_function

from .timer_stats_store import TimerStatsStore
from .constants import ProfileMethod


class DeviceTimer:
    __slots__ = (
        "profiler",
        "timer_stats_store",
        "profiler_function_context",
        "profile_actions",
        "exit_actions",
        "name",
        "aggregation_fn",
        "filter_str",
        "is_disabled",
        "start_event",
        "end_event",
        "start_time",
        "end_time",
        "cpu_only",
    )

    def __init__(
        self,
        name,
        layer_id: int = 0,  # we don't care about layer id, it is just for compatibility with sarathi cudatimer
        profile_method: ProfileMethod = "record_function",
        aggregation_fn=sum,
        filter_str=None,
        cpu_only: bool = False,
    ):
        if name:
            # beautify the names we get from vllm
            name = str(name).replace("OperationMetrics.", "")
            name = name.lower()
            self.name = f"profiler.{name}"
        else:
            self.name = None

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)
        self.is_disabled = (name is None) or self.timer_stats_store.disabled

        if self.is_disabled:
            return

        self.aggregation_fn = aggregation_fn
        self.filter_str = filter_str

        activities = [torch.profiler.ProfilerActivity.CPU]
        if not cpu_only:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        if self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler = torch.profiler.profile(
                activities=activities,
                on_trace_ready=self.handle_trace,
            )
        else:
            self.profiler = None

        self.start_event = None
        self.end_event = None
        self.start_time = None
        self.end_time = None

        self.profile_actions = {
            ProfileMethod.RECORD_FUNCTION: self._start_record_function,
            ProfileMethod.CUDA_EVENT: self._start_cuda_event,
            ProfileMethod.KINETO: self._start_kineto,
            ProfileMethod.PERF_COUNTER: self._start_perf_counter,
        }

        self.exit_actions = {
            ProfileMethod.RECORD_FUNCTION: self._exit_record_function,
            ProfileMethod.CUDA_EVENT: self._exit_cuda_event,
            ProfileMethod.KINETO: self._exit_kineto,
            ProfileMethod.PERF_COUNTER: self._exit_perf_counter,
        }

        self.cpu_only = cpu_only

    def _start_record_function(self):
        self.profiler_function_context = record_function(self.name)
        self.profiler_function_context.__enter__()

    def _start_cuda_event(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def _start_kineto(self):
        self.profiler.__enter__()

    def _start_perf_counter(self):
        if not self.cpu_only:
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()

    # Profiling exit methods
    def _exit_record_function(self, *args):
        self.profiler_function_context.__exit__(*args)

    def _exit_cuda_event(self, *args):
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.end_event.record()
        self.timer_stats_store.record_time(
            self.name, [self.start_event, self.end_event]
        )

    def _exit_kineto(self, *args):
        self.profiler.__exit__(*args)

    def _exit_perf_counter(self, *args):
        if not self.cpu_only:
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.timer_stats_store.record_time(
            self.name,
            (self.end_time - self.start_time) * 1e3,  # convert to ms
        )

    def __enter__(self):
        if self.is_disabled:
            return

        try:
            self.profile_actions[self.timer_stats_store.profile_method]()
        except KeyError:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )

        return self

    def handle_trace(self, trace):
        events = trace.events()

        if self.filter_str:
            events = [e for e in events if e.name.startswith(self.filter_str)]

        total_cuda_time = self.aggregation_fn(
            [
                e.device_time_total if not self.cpu_only else e.cpu_time_total
                for e in events
            ]
        )
        self.timer_stats_store.record_time(
            self.name, total_cuda_time * 1e-3
        )  # convert to ms

    def __exit__(self, *args):
        try:
            self.exit_actions[self.timer_stats_store.profile_method](*args)
        except KeyError:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )
