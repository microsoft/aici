#!/bin/sh

set -x
set -e
mkdir -p src/tokenizers
(cd ../regex_llm && cargo run --release -- -t gpt4 --save ../gvm_use_tok/src/tokenizers/gpt4.bin)
#cargo build --release
#perf stat ./target/release/gvm_use_tok
#exit $?
cargo build --release --target wasm32-unknown-unknown
wasm-opt -Oz target/wasm32-unknown-unknown/release/gvm_use_tok.wasm -o target/opt.wasm
# wasm-strip target/opt.wasm
p=`pwd`
cd ../gvmrt
cargo run --release -- --module $p/target/opt.wasm --run
