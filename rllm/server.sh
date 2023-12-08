#!/bin/sh

REL=--release
REL=

RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=debug \
    cargo run $REL --bin rllm-server -- --verbose --port 7071 "$@"
