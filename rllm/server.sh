#!/bin/sh

REL=--release
REL=

(cd ../aicirt && cargo build --release)

MODEL=codellama/CodeLlama-13b-Instruct-hf
TOK=codellama

RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=trace,aicirt=debug \
    cargo run $REL --bin rllm-server -- \
    --verbose --port 8080 --aicirt ../aicirt/target/release/aicirt \
    --model-id $MODEL --tokenizer $TOK \
    "$@"
