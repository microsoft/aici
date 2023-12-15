#!/bin/sh

set -x
set -e

# the PORT is in fact unused
COMMON_ARGS="--verbose --aicirt ../aicirt/target/release/aicirt"

(cd ../aicirt && cargo build --release)

if [ "X$1" = "X" ] ; then
    HERE=`dirname $0`
    FILES=`echo $HERE/*/args.txt`
else
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
fi


for A in $FILES ; do
    ARGS="$COMMON_ARGS `cat $A`"
    for S in $(dirname $A)/*.safetensors ; do
        ARGS="$ARGS --test $S"
    done
    RUST_BACKTRACE=1 \
    RUST_LOG=info,rllm=debug,aicirt=info \
        cargo run $REL --bin rllm-server -- \
        $ARGS
done

echo "All OK!"
