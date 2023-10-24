#!/bin/sh

T=llama
T=gpt4

set -x
set -e
if test -f tokenizer.bin ; then
  echo "Skipping tokenizer"
else
  (cd ../regex_llm && cargo run --release -- -t $T --save ../aici_ast_runner/tokenizer.bin)
fi
cargo build --release
if [ `uname` = Linux ] ; then
  perf stat ./target/release/aici_ast_runner
else
  ./target/release/aici_ast_runner
fi
