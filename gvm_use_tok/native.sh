#!/bin/sh

T=llama
T=gpt4

set -x
set -e
mkdir -p src/tokenizers
(cd ../regex_llm && cargo run --release -- -t $T --save ../aici_use_tok/tokenizer.bin)
cargo build --release
perf stat ./target/release/aici_use_tok
