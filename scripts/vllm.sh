#!/bin/sh

set -e
set -x
(cd gvm_use_tok && ./wasm.sh)
mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`

RUST_LOG=debug \
PYTHONPATH=.:vllm \
python harness/run_vllm.py \
    --gvm-rt ./gvmrt/target/release/gvmrt \
    --gvm-module $mod \
    --gvm-tokenizer llama \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer
