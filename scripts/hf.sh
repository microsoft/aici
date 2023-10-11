#!/bin/sh

set -e
set -x
(cd aici_use_tok && ./wasm.sh)
mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`

RUST_LOG=debug \
PYTHONPATH=. \
python harness/run_hf.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-module $mod \
    --aici-tokenizer llama \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer
