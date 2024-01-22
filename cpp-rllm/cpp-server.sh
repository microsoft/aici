#!/bin/sh

set -e
REL=
LOOP=
BUILD=
ADD_ARGS=

BIN=$(cd ../target; pwd)

P=`ps -fax|grep 'aicir[t]\|rllm-serve[r]' | awk '{print $1}' | xargs echo`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi

if [ "$1" = bench ] ; then
    REL=--release
    shift
fi

if [ "$1" = cpu ] ; then
    REL="--release --no-default-features"
    shift
fi

case "$1" in
  orca )
    ARGS="-m https://huggingface.co/TheBloke/Orca-2-13B-GGUF/blob/main/orca-2-13b.Q8_0.gguf -t orca"
    ;;
  build )
    BUILD=1
    REL=--release
    ;;
  * )
    echo "try one of models: phi, phi2, 7b, code, code34" 
    exit 1
    ;;
esac
shift

ARGS="--verbose --port 8080 --aicirt $BIN/release/aicirt $ARGS $ADD_ARGS"

(cd ../aicirt; cargo build --release)

cargo build $REL

if [ "$BUILD" = "1" ] ; then
    exit
fi

if [ "X$REL" = "X" ] ; then
    BIN_SERVER=$BIN/debug/cpp-rllm
else
    BIN_SERVER=$BIN/release/cpp-rllm
fi

export RUST_BACKTRACE=1
export RUST_LOG=info,rllm=debug,aicirt=info

echo "running $BIN_SERVER $ARGS $@"

$BIN_SERVER $ARGS "$@"
exit $?
