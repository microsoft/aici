#!/bin/sh

set -x
set -e
mkdir -p src/tokenizers
(cd ../regex_llm && cargo run --release -- -t gpt4 --save ../gvm_use_tok/src/tokenizers/gpt4.bin)
cargo run --release
# wasm-opt -Oz target/wasm32-unknown-unknown/release/gvmprog.wasm -o target/opt.wasm
# wasm-strip target/opt.wasm
# p=`pwd`
# cd ../gvmrt
# cargo run -- --module $p/target/opt.wasm --run
