#!/bin/sh

PORT=8080
REL=--release
REL=

(cd ../aicirt && cargo build --release)

MODEL=microsoft/phi-1_5@refs/pr/18
MODEL=./tmp/phi
TOK=phi

if echo "$*" | grep -q -- --profile-step ; then
    rm -f profile.ncu-rep report1.*
    cargo build --release
RUST_LOG=info \
    nsys profile -c cudaProfilerApi \
    --stats true \
    ./target/release/rllm-server \
    --verbose --port $PORT --aicirt ../aicirt/target/release/aicirt \
    --model $MODEL --tokenizer $TOK \
    "$@"
else
RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=debug,aicirt=info \
    cargo run $REL --bin rllm-server -- \
    --verbose --port $PORT --aicirt ../aicirt/target/release/aicirt \
    --model $MODEL --tokenizer $TOK \
    "$@"
fi
