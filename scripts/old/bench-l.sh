#!/bin/sh

PYTHONPATH=. python benchmarks/benchmark_latency.py --model NousResearch/Llama-2-13b-chat-hf --tokenizer hf-internal-testing/llama-tokenizer

# Avg latency: 2.836 seconds