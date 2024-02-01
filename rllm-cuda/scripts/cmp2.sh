#!/bin/sh

set -x
set -e
cargo run -- --sample-len 1
mv step-1.safetensor single.safetensor
cargo run -- --sample-len 1 --alt=7
mv step-1.safetensor multi.safetensor
python3 tensorcmp.py single.safetensor multi.safetensor
