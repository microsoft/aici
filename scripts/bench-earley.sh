#!/bin/sh

set -x
set -e
(cd aicirt && cargo build --release)
perf stat ./target/release/aicirt --tokenizer gpt4 --earley-bench
