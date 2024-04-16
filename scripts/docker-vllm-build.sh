#!/bin/sh

cd py/vllm
DOCKER_BUILDKIT=1 \
docker build . \
    --target vllm-openai \
    --tag vllm/vllm-openai \
    --build-arg max_jobs=16 \
    --build-arg nvcc_threads=16 \
    --build-arg torch_cuda_arch_list="8.0"
