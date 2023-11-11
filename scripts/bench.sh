#!/bin/sh

set -e
set -x
(cd aicirt && cargo run --release -- --bench)

PYTHONPATH=.:vllm \
python harness/bench.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama
