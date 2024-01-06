#!/bin/sh
PROMPT="$*"
if [ -z "$PROMPT" ]; then
    PROMPT="Is coffee any good?"
fi
set -x
(cd aici_abi &&  cargo build --release) && \
    ./scripts/upload.sh --vm target/wasm32-wasi/release/yesno.wasm --prompt "$PROMPT"
