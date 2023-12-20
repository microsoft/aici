#!/bin/sh

set -e
REL=
LOOP=

if [ "$1" = loop ] ; then
    REL=--release
    LOOP=1
    shift
fi

case "$1" in
  phi )
    ARGS="-m microsoft/phi-1_5@refs/pr/66 -t phi -w expected/phi-1_5/cats.safetensors"
    ;;
  phi2 )
    ARGS="-m microsoft/phi-2 -t phi -w expected/phi-2/cats.safetensors"
    ;;
  7 | 7b )
    ARGS="-m NousResearch/Llama-2-7b-hf -t llama -w expected/llama/cats.safetensors"
    ;;
  code )
    ARGS="-m codellama/CodeLlama-13b-Instruct-hf -t llama16 -w expected/codellama/cats.safetensors"
    ;;
  code34 )
    ARGS="-m codellama/CodeLlama-34b-Instruct-hf -t llama"
    ;;
  * )
    echo "try one of models: phi, phi2, 7b, code, code34" 
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
    echo $?
fi

cargo build $REL --bin rllm-server
if [ "X$REL" = "X" ] ; then
    BIN=./target/debug/rllm-server
else
    BIN=./target/release/rllm-server
fi

export RUST_BACKTRACE=1
export RUST_LOG=info,rllm=debug,aicirt=info

echo "running $BIN $ARGS $@"

if [ "$LOOP" = "" ] ; then
    $BIN $ARGS "$@"
    exit $?
fi

set +e
while : ; do
    $BIN --daemon $ARGS "$@" 2>&1 | rotatelogs -e -D ./logs/%Y-%m-%d-%H_%M_%S.txt 3600 
    echo "restarting..."
    sleep 2
done
