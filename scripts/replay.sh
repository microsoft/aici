#!/bin/sh

set -e
set -x

(cd aicirt && cargo build --release)

f="$1"
if [ "X$f" = "X" ]; then
    f=tmp/trace.jsonl
fi

RUST_LOG=info \
PYTHONPATH=.:vllm \
python harness/replay.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama \
    "$f"
