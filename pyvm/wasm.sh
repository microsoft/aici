#!/bin/sh

set -x
set -e
# (cd ../aicirt && cargo build --release)
TRG=wasm32-wasi
cargo build --release --target $TRG
cp target/$TRG/release/aici_pyvm.wasm target/opt.wasm
wasm-strip -k name target/opt.wasm
ls -l target/opt.wasm
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
if [ "X$1" = "Xcache" ] ; then
  ./aicirt/target/release/aicirt --module $p/target/opt.wasm | tee tmp/runlog.txt
  exit
fi

PERF=
if [ `uname` = Linux ] ; then
  PERF="perf stat"
fi
RUST_LOG=info $PERF ./aicirt/target/release/aicirt --tokenizer gpt4 --module $p/target/opt.wasm
RUST_LOG=info $PERF ./aicirt/target/release/aicirt \
  --tokenizer gpt4 --module $p/target/opt.wasm --run | tee tmp/runlog.txt
ls -l $p/target/opt.wasm
