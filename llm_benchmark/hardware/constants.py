from enum import Enum


class FlopsPerCycle(float, Enum):
    AMX = 1024.0
    AVX512 = 32.0
    AVX2 = 16.0
    AVX = 8.0
    DEFAULT = 4.0