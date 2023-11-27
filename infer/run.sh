#!/bin/sh
RUST_BACKTRACE=1 \
RUST_LOG=debug \
    cargo run -- --sample-len 10 "$@"
