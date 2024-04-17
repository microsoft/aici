#!/bin/sh

set -e
set -x

if [ -z "$FOLDER" ]; then
    MODEL_ARGS="--model microsoft/Orca-2-13b --revision refs/pr/22 --aici-tokenizer orca"
else
    MODEL_ARGS="--model ./$FOLDER --aici-tokenizer ./$FOLDER/tokenizer.json --tokenizer ./$FOLDER"
fi

(cd aicirt && cargo build --release)

RUST_LOG=info,tokenizers=error,aicirt=trace \
PYTHONPATH=py:py/vllm \
python3 -m vllm.entrypoints.openai.api_server \
    --enforce-eager \
    --use-v2-block-manager \
    --enable-chunked-prefill \
    --aici-rt ./target/release/aicirt \
    $MODEL_ARGS \
    --port 4242 --host 127.0.0.1 \
    "$@"

#    --aici-rtarg="--wasm-max-step-time=50" \
#    --aici-rtarg="--wasm-max-pre-step-time=2" \
#    --aici-rtarg="--wasm-max-init-time=1000" \
#    --aici-rtarg="--wasm-max-memory=64" \
#    --aici-rtarg="--wasm-max-pre-step-time=10" \
