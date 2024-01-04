#!/bin/sh

set -x
set -e
# (cd ../aicirt && cargo build --release)
cargo build --release --target wasm32-wasi
BIN=$(cd ../target; pwd)
cp $BIN/wasm32-wasi/release/aici_ast_runner.wasm $BIN/opt.wasm
wasm-strip -k name $BIN/opt.wasm # -o $BIN/strip.wasm
ls -l $BIN/opt.wasm
# curl -X POST -T "$BIN/opt.wasm" "http://127.0.0.1:8080/v1/aici_modules"
if [ "X$1" = "Xbuild" ] ; then
  exit
fi
if [ "X$1" = "Xsize" ] ; then
  node size.js
  fx $BIN/dominators.json
  exit
fi

cargo build --release --package aicirt
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
