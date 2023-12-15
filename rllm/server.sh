#!/bin/sh

REL=--release
REL=

case "$1" in
  phi )
    MODEL=microsoft/phi-1_5@refs/pr/18
    MODEL=./tmp/phi
    TOK=phi
    ;;
  7 | 7b )
    MODEL="NousResearch/Llama-2-7b-hf"
    TOK=llama
    ;;
  code )
    MODEL=codellama/CodeLlama-13b-Instruct-hf
    TOK=codellama
    ;;
  * )
    echo "try one of models: phi, 7b, code" 
    ;;
esac
shift

ARGS="--verbose --port 8080 --aicirt ../aicirt/target/release/aicirt"
ARGS="$ARGS --model $MODEL --tokenizer $TOK"

(cd ../aicirt && cargo build --release)

if echo "$*" | grep -q -- --profile-step ; then
    rm -f profile.ncu-rep report1.*
    cargo build --release
RUST_LOG=info \
    nsys profile -c cudaProfilerApi \
    --stats true \
    ./target/release/rllm-server \
    $ARGS "$@"
else
RUST_BACKTRACE=1 \
RUST_LOG=info,rllm=debug,aicirt=info \
    cargo run $REL --bin rllm-server -- \
    $ARGS "$@"
fi
