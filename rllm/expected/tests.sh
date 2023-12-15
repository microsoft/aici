#!/bin/sh

set -x
set -e

# the PORT is in fact unused
PORT=8080
COMMON_ARGS="--verbose --port $PORT --aicirt ../aicirt/target/release/aicirt"

(cd ../aicirt && cargo build --release)

HERE=`dirname $0`

for A in $HERE/*/args.txt ; do
    ARGS="$COMMON_ARGS `cat $A`"
    for S in $(dirname $A)/*.safetensors ; do
        ARGS="$ARGS --test $S"
    done
    RUST_BACKTRACE=1 \
    RUST_LOG=info,rllm=debug,aicirt=info \
        cargo run $REL --bin rllm-server -- \
        $ARGS "$@"
done

echo "All OK!"
