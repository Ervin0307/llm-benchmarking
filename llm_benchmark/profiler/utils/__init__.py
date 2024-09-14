import enum


class ProfileMethod(enum.Enum):
    DEVICE_EVENT = "device_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"
