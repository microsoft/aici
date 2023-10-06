#!/bin/sh

PYTHONPATH=. python benchmarks/benchmark_throughput.py --dataset benchmarks/small.json --num-prompts 200 --model NousResearch/Llama-2-13b-chat-hf --tokenizer hf-internal-testing/llama-tokenizer
# Throughput: 4.12 requests/s, 2110.03 tokens/s, 858.21 out tokens/s