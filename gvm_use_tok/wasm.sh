#!/bin/sh

set -x
set -e
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/gvm_use_tok.wasm target/opt.wasm
# wasm-opt -Oz target/wasm32-unknown-unknown/release/gvm_use_tok.wasm -o target/opt.wasm
# wasm-strip target/opt.wasm
p=`pwd`
cd ../gvmrt
cargo build --release
cd ..
mkdir -p tmp
./gvmrt/target/release/gvmrt --tokenizer gpt4 --module $p/target/opt.wasm --run | tee tmp/runlog.txt
ls -l $p/target/opt.wasm