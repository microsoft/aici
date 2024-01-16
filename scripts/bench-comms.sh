#!/bin/sh

set -x
./scripts/kill-rt.sh

set -e
(cd aicirt && cargo run --release -- --bench --name /aicibench-)

./aici.sh benchrt \
    --aici-rt ./aicirt/target/release/aicirt \
    --aici-tokenizer llama
