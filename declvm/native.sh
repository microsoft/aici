#!/bin/sh

T=llama
T=gpt4

set -x
set -e
if test -f tokenizer.bin ; then
  echo "Skipping tokenizer"
else
  (cd ../aicirt && cargo run --release -- --tokenizer $T --save-tokenizer ../declvm/tokenizer.bin)
fi
cargo build --release
if [ `uname` = Linux ] ; then
  perf stat ./target/release/declvm
else
  ./target/release/declvm
fi
