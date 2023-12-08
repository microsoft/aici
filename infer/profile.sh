#!/bin/sh

set -e
set -x

rm -f profile.ncu-rep report1.*
cargo build --release

RUST_LOG=info,rllm=trace \
nsys profile -c cudaProfilerApi \
    --stats true \
    ./target/release/infer --nv-profile

set +x
echo "to repeat: nsys stats report1.nsys-rep"

# RUST_LOG=info,rllm=trace \
# ncu \
#     -s 299 \
#     -o profile \
#     ./target/debug/infer --sample-len=2
