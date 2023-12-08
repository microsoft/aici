#!/bin/sh
RUST_LOG=info,rllm=trace \
RUST_BACKTRACE=1 \
cargo test "$@"
