#!/bin/sh

set -e
REL=--release
LOOP=
BUILD=
ADD_ARGS=

mkdir -p ../target
BIN=$(cd ../target; pwd)

if [ -f "../tch-cuda/cutlass/README.md" ] ; then
  :
else
  (cd .. && git submodule update --init --recursive)
fi

P=`ps -ax|grep 'aicir[t]\|rllm-serve[r]' | awk '{print $1}' | xargs echo`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi

if [ "$1" = "--loop" ] ; then
    LOOP=1
    shift
fi

if [ "$1" = "--debug" ] ; then
    REL=
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
    # OOM in hf transformers - can't generate testcases...
    ARGS="-m codellama/CodeLlama-34b-Instruct-hf -t llama"
    ;;
  orca )
    ARGS="-m microsoft/Orca-2-13b@refs/pr/22 -t orca -w expected/orca/cats.safetensors"
    ;;
  build )
    BUILD=1
    REL=--release
    ;;
  * )
    echo "usage: $0 [--loop] [--debug] [phi|phi2|7b|code|orca|build] [rllm_args...]"
    echo "Try $0 phi2 --help to see available rllm_args"
    exit 1
    ;;
esac
shift

ARGS="--verbose --port 8080 --aicirt $BIN/release/aicirt $ARGS $ADD_ARGS"

(cd ../aicirt; cargo build --release)

if echo "$*" | grep -q -- --profile-step ; then
    rm -f profile.ncu-rep report1.*
    cargo build --release
RUST_LOG=info \
    nsys profile -c cudaProfilerApi \
    $BIN/rllm-server \
    $ARGS "$@"
    nsys stats ./report1.nsys-rep > tmp/perf.txt
    echo "Opening tmp/perf.txt in VSCode; use Alt-Z to toggle word wrap"
    code tmp/perf.txt
    exit
fi

cargo build $REL --bin rllm-server

if [ "$BUILD" = "1" ] ; then
    exit
fi

if [ "X$REL" = "X" ] ; then
    BIN_SERVER=$BIN/debug/rllm-server
else
    BIN_SERVER=$BIN/release/rllm-server
fi

if [ `uname` = Darwin ] ; then
  mkdir -p tmp
  TP=tmp/torch-path.txt
  if test -f $TP ; then
    :
  else
    python -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])" > $TP
  fi
  TP=`cat $TP`
  echo "Updating torch RPATH: $TP on $BIN_SERVER"
  install_name_tool -add_rpath $TP $BIN_SERVER
fi

export RUST_BACKTRACE=1
export RUST_LOG=info,rllm=debug,aicirt=info

echo "running $BIN_SERVER $ARGS $@"

if [ "$LOOP" = "" ] ; then
    $BIN_SERVER $ARGS "$@"
    exit $?
fi

set +e
while : ; do
    $BIN_SERVER --daemon $ARGS "$@" 2>&1 | rotatelogs -e -D ./logs/%Y-%m-%d-%H_%M_%S.txt 3600 
    echo "restarting..."
    sleep 2
done
