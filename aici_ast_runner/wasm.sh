#!/bin/sh

set -x
set -e
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/aici_ast_runner.wasm target/opt.wasm
# wasm-opt -Oz target/wasm32-unknown-unknown/release/aici_ast_runner.wasm -o target/opt.wasm
wasm-strip target/opt.wasm
# curl -X POST -T "target/opt.wasm" "http://127.0.0.1:8080/v1/aici_modules"
exit

p=`pwd`
cd ../aicirt
cargo build --release
cd ..
mkdir -p tmp
./aicirt/target/release/aicirt --tokenizer gpt4 --module $p/target/opt.wasm --run | tee tmp/runlog.txt
ls -l $p/target/opt.wasm