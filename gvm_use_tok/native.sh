#!/bin/sh

set -x
set -e
mkdir -p src/tokenizers
(cd ../regex_llm && cargo run --release -- -t gpt4 --save ../gvm_use_tok/tokenizer.bin)
cargo build --release
perf stat ./target/release/gvm_use_tok
