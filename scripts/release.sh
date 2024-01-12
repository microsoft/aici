#!/bin/sh

FOLDERS="aici_abi uppercase pyctrl jsctrl declctrl aicirt"
NATIVE="$(uname -s | tr 'A-Z' 'a-z')-$(uname -m)"
D=`date +%Y%m%d-%H%M`
TAG=`git describe --dirty --tags --match 'v[0-9]*' --always | sed -e 's/^v//; s/-dirty/-'"$D/"`
XZ=

if [ "$1" == "--xz" ] ; then
    XZ=1
    shift
fi

echo "Building for $NATIVE"

set -e

for f in $FOLDERS ; do
    echo "Build $f..."
    (cd $f && cargo build --release)
done

function release() {

T0=$1
T=target/dist/$1
shift

TITLE="$1"
shift

SUFF="$1"
shift

rm -rf $T
mkdir -p $T
echo "# $TITLE ($SUFF-$TAG)" > $T/README.md

BN=
for f in "$@" ; do
    cp $f $T/
    BN="$BN $(basename $f)"
done

echo >> $T/README.md
echo "Contents:" >> $T/README.md
echo '```' >> $T/README.md
(cd $T && ls -l $BN | awk '{print $5, $9}') >> $T/README.md
echo >> $T/README.md
(cd $T && sha256sum $BN) >> $T/README.md
echo '```' >> $T/README.md
echo >> $T/README.md

cat $T/README.md >> target/dist/README.md

if [ "$XZ" = "1" ] ; then
DST=target/dist/$T0-$SUFF-$TAG.tar.xz
tar -Jcf $DST -C target/dist $T0
ls -l $DST
fi

DST=target/dist/$T0-$SUFF-$TAG.tar.gz
tar -zcf $DST -C target/dist $T0
ls -l $DST

}

rm -rf target/dist
mkdir -p target/dist
echo -n > target/dist/README.md

release aici-controllers "AICI Controllers" "wasm32-wasi" target/wasm32-wasi/release/*.wasm
strip target/release/aicirt
release aicirt "AICI Runtime" "$NATIVE" target/release/aicirt
