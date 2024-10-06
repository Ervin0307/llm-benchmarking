#!/bin/bash

PROFILER_RESULT_DIR=/datadrive_8T/.cache/profiling \
    python auto_benchmark.py --port 8989 --engine-config-file example/vllm_engine_gpu.yaml --docker-image "vllm/cuda:12.4.1" \
        --run-benchmark --profile-collectives --profile-hardware > run.log 2>&1
