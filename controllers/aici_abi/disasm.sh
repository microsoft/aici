#!/bin/sh

TRG=`rustup show | head -1 | sed -e 's/.*: //'`
CRATE=`grep "^name =" Cargo.toml  | head -1 | sed -e 's/.*= "//; s/"//'`
RUSTFLAGS="--emit asm" cargo build --release --target $TRG
F=`echo ../../target/$TRG/release/deps/$CRATE-*.s`
# if $F has more than one file
if [ `echo $F | wc -w` -gt 1 ]; then
    echo "More than one file found: $F; removing; try again"
    rm -f $F
    exit 1
fi

mkdir -p tmp
cp $F tmp/full.s
node annotate_asm.js tmp/full.s "$@" | rustfilt > tmp/func.s
ls -l tmp/func.s
