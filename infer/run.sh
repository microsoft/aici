#!/bin/sh
RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=debug \
    cargo run -- --sample-len 10 "$@"
