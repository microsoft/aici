#!/bin/sh

set -e
set -x

(cd aicirt && cargo build --release)

RUST_LOG=info \
PYTHONPATH=.:vllm \
python harness/replay.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama \
    tmp/trace.jsonl
