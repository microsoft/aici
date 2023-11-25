#!/bin/sh
RUST_BACKTRACE=1 \
    cargo run -- --sample-len 10 "$@"
