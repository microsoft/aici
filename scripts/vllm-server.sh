#!/bin/sh

set -e
set -x

MODEL=NousResearch/Llama-2-7b-chat-hf
TOK=llama

#MODEL=codellama/CodeLlama-34b-Instruct-hf
MODEL=codellama/CodeLlama-13b-Instruct-hf
TOK=llama16

(cd aicirt && cargo build --release)

RUST_LOG=info \
PYTHONPATH=py:py/vllm \
python3 scripts/py/vllm_server.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer $TOK \
    --aici-trace tmp/trace.jsonl \
    --model $MODEL \
    --aici-rtarg="--wasm-max-pre-step-time=10" \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --port 4242 --host 127.0.0.1

#    --aici-rtarg="--wasm-max-step-time=50" \
#    --aici-rtarg="--wasm-max-pre-step-time=2" \
#    --aici-rtarg="--wasm-max-init-time=1000" \
#    --aici-rtarg="--wasm-max-memory=64" \
