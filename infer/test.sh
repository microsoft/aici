#!/bin/sh
RUST_BACKTRACE=1 \
cargo test "$@"
