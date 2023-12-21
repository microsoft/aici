#!/bin/sh

set -e

# the PORT is in fact unused
COMMON_ARGS="--verbose --aicirt ../aicirt/target/release/aicirt"

(cd ../aicirt && cargo build --release)

RLLM_LOG=debug

FILES=
for f in "$@" ; do
    if [ -f "$f" ] ; then
        FILES="$FILES $f"
    elif [ -f "$f/args.txt" ] ; then
        FILES="$FILES $f/args.txt"
    else
        echo "File $f not found"
        exit 1
    fi
done

for A in $FILES ; do
    echo
    echo
    echo
    echo "*** $A ***"
    echo
    ARGS="$COMMON_ARGS `cat $A`"
    for S in $(dirname $A)/*.safetensors ; do
        ARGS="$ARGS --test $S"
    done
    RUST_BACKTRACE=1 \
    RUST_LOG=info,rllm=$RLLM_LOG,aicirt=info \
        cargo run $REL --bin rllm-server -- \
        $ARGS
done

echo "All OK!"
