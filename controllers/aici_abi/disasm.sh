#!/bin/sh

RUSTFLAGS="--emit asm" cargo build --release --target x86_64-unknown-linux-gnu
F=`echo ../../target/x86_64-unknown-linux-gnu/release/deps/aici_abi-*.s`
# if $F has more than one file
if [ `echo $F | wc -w` -gt 1 ]; then
    echo "More than one file found: $F; removing; try again"
    rm -f $F
    exit 1
fi

mkdir -p tmp

rustfilt < $F > tmp/aici_abi.s
