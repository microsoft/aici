#!/bin/sh
(cd aici_abi &&  cargo build --release) && \
    ./scripts/upload.sh --vm target/wasm32-wasi/release/uppercase.wasm