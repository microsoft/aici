#!/bin/sh

set -e
set -x
(cd declctrl && ./wasm.sh cache)
mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`

RUST_LOG=info \
PYTHONPATH=py:py/vllm \
python3 scripts/py/run_vllm.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-module $mod \
    --aici-module-arg declctrl/arg2.json \
    --aici-tokenizer llama \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer
