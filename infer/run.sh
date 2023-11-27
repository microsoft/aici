#!/bin/sh
RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=trace \
    cargo run -- --sample-len 10 "$@"
