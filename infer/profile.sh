#!/bin/sh

set -e
set -x

rm -f profile.ncu-rep
cargo build
RUST_LOG=info,rllm=trace \
ncu \
    -s 299 \
    -o profile \
    ./target/debug/infer --sample-len=2
