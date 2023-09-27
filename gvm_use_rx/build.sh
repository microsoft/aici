#!/bin/sh

set -x
set -e
(cd ../regex_llm && cargo run --release)
cargo build --release
wasm-opt -Oz target/wasm32-unknown-unknown/release/gvmprog.wasm -o target/opt.wasm
wasm-strip target/opt.wasm
p=`pwd`
cd ../gvmrt
cargo run -- --module $p/target/opt.wasm --run
