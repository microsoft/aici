#!/bin/sh

set -e
set -x
#(cd aici_ast_runner && ./wasm.sh)
#mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`

RUST_LOG=debug \
PYTHONPATH=.:vllm \
python harness/vllm_server.py \
    --model NousResearch/Llama-2-7b-chat-hf \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --port 8080 --host 127.0.0.1