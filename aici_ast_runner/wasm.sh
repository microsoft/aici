#!/bin/sh

set -x
set -e
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/aici_ast_runner.wasm target/opt.wasm
# wasm-opt -Oz target/wasm32-unknown-unknown/release/aici_ast_runner.wasm -o target/opt.wasm
# wasm-strip target/opt.wasm
p=`pwd`
cd ../aicirt
cargo build --release
cd ..
mkdir -p tmp
./aicirt/target/release/aicirt --tokenizer gpt4 --module $p/target/opt.wasm --run | tee tmp/runlog.txt
ls -l $p/target/opt.wasm