from enum import Enum


class FlopsPerCycle(Enum):
    AMX = 1024
    AVX512 = 32
    AVX2 = 16
    AVX = 8
    DEFAULT = 4