#!/bin/sh

set -x
set -e
# (cd ../aicirt && cargo build --release)
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/aici_ast_runner.wasm target/opt.wasm
# wasm-opt -Oz target/wasm32-unknown-unknown/release/aici_ast_runner.wasm -o target/opt.wasm
wasm-strip -k name target/opt.wasm -o target/strip.wasm
ls -l target/strip.wasm
# curl -X POST -T "target/opt.wasm" "http://127.0.0.1:8080/v1/aici_modules"
if [ "X$1" = "Xbuild" ] ; then
  exit
fi
if [ "X$1" = "Xsize" ] ; then
  node size.js
  fx target/dominators.json
  exit
fi

p=`pwd`
cd ../aicirt
cargo build --release
cd ..
mkdir -p tmp
PERF=
if [ `uname` = Linux ] ; then
  PERF="perf stat"
fi
RUST_LOG=info $PERF ./aicirt/target/release/aicirt --tokenizer gpt4 --module $p/target/opt.wasm
RUST_LOG=info $PERF ./aicirt/target/release/aicirt \
  --tokenizer gpt4 --module $p/target/opt.wasm --run | tee tmp/runlog.txt
ls -l $p/target/opt.wasm $p/target/strip.wasm
