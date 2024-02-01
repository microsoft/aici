#!/bin/sh

echo "This is outdated."
exit 1

set -e
set -x
(cd declctrl && ./wasm.sh cache)
mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`

RUST_LOG=debug \
PYTHONPATH=. \
python3 harness/run_hf.py \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-module $mod \
    --aici-module-arg declctrl/arg.json \
    --aici-tokenizer llama \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer
