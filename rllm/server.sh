#!/bin/sh

REL=--release
REL=

(cd ../aicirt && cargo build --release)

MODEL=codellama/CodeLlama-13b-Instruct-hf
TOK=codellama

if echo "$*" | grep -q -- --profile-step ; then
    rm -f profile.ncu-rep report1.*
    cargo build --release
RUST_LOG=info \
    nsys profile -c cudaProfilerApi \
    --stats true \
    ./target/release/rllm-server \
    --verbose --port 8080 --aicirt ../aicirt/target/release/aicirt \
    --model-id $MODEL --tokenizer $TOK \
    "$@"
else
RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=debug,aicirt=info \
    cargo run $REL --bin rllm-server -- \
    --verbose --port 8080 --aicirt ../aicirt/target/release/aicirt \
    --model-id $MODEL --tokenizer $TOK \
    "$@"
fi




