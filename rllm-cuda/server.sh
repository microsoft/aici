#!/bin/bash

set -e
WS=`cd $(dirname $0)/.. && pwd`
REL=--release
LOOP=
BUILD=
ADD_ARGS=
R_LOG=info,tokenizers=error,rllm=debug,aicirt=info


mkdir -p "$WS/target"
BIN="$WS/target"

if [ -f "$WS/tch-cuda/cutlass/README.md" ] ; then
  :
else
  (cd $WS && git submodule update --init --recursive)
fi

if [ "X$CUDA_VISIBLE_DEVICES" = "X" ] ; then
  P=`ps -ax|grep 'aicir[t]\|rllm-serve[r]|rll[m]-cpp' | awk '{print $1}' | xargs echo`
  if [ "X$P" != "X" ] ; then 
    echo "KILL $P"
    kill $P
  fi
else
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ "$CPP" = 1 ] ; then
  VER="--no-default-features"
else
  VER=
fi

if [ "$1" = "--trace" ] ; then
    R_LOG=info,rllm=trace,aicirt=info
    shift
fi

if [ "$1" = "--trace-rt" ] ; then
    R_LOG=info,rllm=trace,aicirt=trace
    shift
fi

if [ "$1" = "--loop" ] ; then
    LOOP=1
    shift
fi

if [ "$1" = "--cuda" ] ; then
    if [ "$CPP" = 1 ] ; then
      VER="$VER --features cuda"
      ADD_ARGS="--gpu-layers 1000"
    else
       echo "--cuda only valid for llama.cpp"
       exit 1
    fi
    shift
fi

if [ "$1" = "--debug" ] ; then
    REL=
    shift
fi

EXPECTED=$WS/rllm-cuda/expected

if [ "$CPP" = 1 ] ; then
  BIN_NAME=rllm-cpp
  FOLDER_NAME=rllm-cpp
  case "$1" in
    phi2 )
      ARGS="-m https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q8_0.gguf -t phi -w $EXPECTED/phi-2/cats.safetensors -s test_maxtol=0.8 -s test_avgtol=0.3"
      ;;
    orca )
      ARGS="-m https://huggingface.co/TheBloke/Orca-2-13B-GGUF/blob/main/orca-2-13b.Q8_0.gguf -t orca -w $EXPECTED/orca/cats.safetensors"
      ;;
    mistral )
      ARGS="-m https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf"
      ;;
    mixtral )
      ARGS="-m https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/blob/main/mixtral-8x7b-instruct-v0.1.Q6_K.gguf"
      ;;
    https* )
      ARGS="-m $1"
      ;;
    build )
      BUILD=1
      ;;
    * )
      echo "usage: $0 [--loop] [--cuda] [--debug] [phi2|orca|build] [rllm_args...]"
      echo "Try $0 phi2 --help to see available rllm_args"
      exit 1
      ;;
  esac
else
  BIN_NAME=rllm-server
  FOLDER_NAME=rllm-cuda
  case "$1" in
    phi )
      ARGS="-m microsoft/phi-1_5@refs/pr/66 -t phi -w $EXPECTED/phi-1_5/cats.safetensors"
      ;;
    phi2 )
      ARGS="-m microsoft/phi-2@d3186761bf5c4409f7679359284066c25ab668ee -t phi -w $EXPECTED/phi-2/cats.safetensors"
      ;;
    7 | 7b )
      ARGS="-m NousResearch/Llama-2-7b-hf -t llama -w $EXPECTED/llama/cats.safetensors"
      ;;
    code )
      ARGS="-m codellama/CodeLlama-13b-Instruct-hf -t llama16 -w $EXPECTED/codellama/cats.safetensors"
      ;;
    code34 )
      # OOM in hf transformers - can't generate testcases...
      ARGS="-m codellama/CodeLlama-34b-Instruct-hf -t llama"
      ;;
    orca )
      ARGS="-m microsoft/Orca-2-13b@refs/pr/22 -t orca -w $EXPECTED/orca/cats.safetensors"
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
fi

shift

ARGS="--verbose --aicirt $BIN/release/aicirt $ARGS $ADD_ARGS"

(cd $WS/aicirt && cargo build --release)
(cd $WS/$FOLDER_NAME && cargo build $REL $VER)

if [ "X$REL" = "X" ] ; then
    BIN_SERVER=$BIN/debug/$BIN_NAME
else
    BIN_SERVER=$BIN/release/$BIN_NAME
fi

if echo "$*" | grep -q -- --profile-step ; then
    rm -f profile.ncu-rep report1.*
    cargo build --release
RUST_LOG=info \
    nsys profile -c cudaProfilerApi \
    $BIN_SERVER \
    $ARGS "$@"
    nsys stats ./report1.nsys-rep > tmp/perf.txt
    echo "Opening tmp/perf.txt in VSCode; use Alt-Z to toggle word wrap"
    code tmp/perf.txt
    exit
fi


if [ "$BUILD" = "1" ] ; then
    exit
fi

if [ "$CPP" != "1" ] && [ `uname` = Darwin ] ; then
  mkdir -p tmp
  TP=tmp/torch-path.txt
  if test -f $TP ; then
    :
  else
    python3 -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])" > $TP
  fi
  TP=`cat $TP`
  echo "Updating torch RPATH: $TP on $BIN_SERVER"
  install_name_tool -add_rpath $TP $BIN_SERVER
fi

export RUST_BACKTRACE=1
export RUST_LOG=$R_LOG

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
