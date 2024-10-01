#!/bin/bash

PROFILER_RESULT_DIR=/datadrive_8T/.cache/profiling \
    python auto_benchmark.py --model meta-llama/Meta-Llama-3-8B-Instruct --port 8989 \
        --engine-config-file example/vllm_engine_gpu.yaml --docker-image "vllm/cuda:12.4.1" \
        --input-tokens 128,256,512,1024,2048,4096 --output-tokens 128,256,512,1024,2048 --concurrency 1,10,20,30,50,100,150,200,250,300 \
        --run-benchmark --profile-collectives --profile-hardware 2>&1 | tee run.log