#!/bin/sh

set -x
set -e
cargo build --release
BIN=$(cd ../target; pwd)
cp $BIN/wasm32-wasi/release/aici_ast_runner.wasm $BIN/opt.wasm
ls -l $BIN/opt.wasm
if [ "X$1" = "Xbuild" ] ; then
  exit
fi
if [ "X$1" = "Xsize" ] ; then
  node size.js
  fx $BIN/dominators.json
  exit
fi

(cd ../aicirt; cargo build --release)

mkdir -p tmp
if [ "X$1" = "Xcache" ] ; then
  $BIN/release/aicirt --module $BIN/opt.wasm | tee tmp/runlog.txt
  exit
fi

PERF=
if [ `uname` = Linux ] ; then
  PERF="perf stat"
fi
RUST_LOG=info $PERF $BIN/release/aicirt --tokenizer gpt4 --module $BIN/opt.wasm
RUST_LOG=info $PERF $BIN/release/aicirt \
  --tokenizer gpt4 --module $BIN/opt.wasm --run | tee tmp/runlog.txt
ls -l $BIN/opt.wasm
