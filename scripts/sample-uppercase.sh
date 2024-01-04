#!/bin/sh
(cd aici_abi &&  cargo build --release) && ./scripts/upload.sh target/wasm32-wasi/release/uppercase.wasm