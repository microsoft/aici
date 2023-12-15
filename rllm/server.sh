#!/bin/sh

REL=--release
REL=

case "$1" in
  phi )
    # microsoft/phi-1_5@refs/pr/18
    ARGS="-m ./tmp/phi -t phi -w expected/phi-1_5/cats.safetensors"
    ;;
  7 | 7b )
    ARGS="-m NousResearch/Llama-2-7b-hf -t llama"
    ;;
  code )
    ARGS="-m codellama/CodeLlama-13b-Instruct-hf -t codellama"
    ;;
  * )
    echo "try one of models: phi, 7b, code" 
    exit 1
    ;;
esac
shift

ARGS="--verbose --port 8080 --aicirt ../aicirt/target/release/aicirt $ARGS"

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
