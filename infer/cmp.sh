#!/bin/sh
set -e
set -x
export RUST_BACKTRACE=1
cargo build
mkdir -p tmp
cargo run -- --sample-len 5 "$@" --reference > tmp/reference.txt
cargo run -- --sample-len 5 "$@"  > tmp/new.txt
diff -u tmp/reference.txt tmp/new.txt > tmp/ref.diff || true
