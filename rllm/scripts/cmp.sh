#!/bin/sh
K=3
set -e
set -x
export RUST_BACKTRACE=1
cargo build
mkdir -p tmp
cargo run -- --sample-len $K "$@"  > tmp/reference.txt
cargo run -- --sample-len $K --alt=1 "$@"  > tmp/new.txt
diff -u200 tmp/reference.txt tmp/new.txt > tmp/diff.patch || true
ls -l tmp/diff.patch
