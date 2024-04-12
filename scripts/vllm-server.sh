#!/bin/sh

set -e
set -x

MODEL="microsoft/Orca-2-13b"
MODEL_REV="refs/pr/22"
AICI_TOK=orca

(cd aicirt && cargo build --release)

RUST_LOG=info,tokenizers=error,aicirt=trace \
PYTHONPATH=py:py/vllm \
python3 -m vllm.entrypoints.openai.api_server \
    --enforce-eager \
    --use-v2-block-manager \
    --enable-chunked-prefill \
    --aici-rt ./target/release/aicirt \
    --aici-tokenizer $AICI_TOK \
    --model $MODEL \
    --revision $MODEL_REV \
    --port 4242 --host 127.0.0.1

#    --aici-rtarg="--wasm-max-step-time=50" \
#    --aici-rtarg="--wasm-max-pre-step-time=2" \
#    --aici-rtarg="--wasm-max-init-time=1000" \
#    --aici-rtarg="--wasm-max-memory=64" \
#    --aici-rtarg="--wasm-max-pre-step-time=10" \
