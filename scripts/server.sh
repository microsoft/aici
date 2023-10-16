#!/bin/sh

set -e
set -x

(cd aicirt && cargo build --release)

RUST_LOG=info \
PYTHONPATH=.:vllm \
python harness/vllm_server.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama \
    --aici-trace tmp/trace.jsonl \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --port 8080 --host 127.0.0.1