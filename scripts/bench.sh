#!/bin/sh

set -x
./scripts/kill-rt.sh

set -e
(cd aicirt && cargo run --release -- --bench --name /aicibench-)

X_RUST_LOG=debug \
PYTHONPATH=.:vllm \
python harness/bench.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama
