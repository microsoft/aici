#!/bin/bash
#
# This is run from the GitHub Action, but can be also run locally to see if everything builds correctly.
#

FOLDERS="
controllers/aici_abi
controllers/declctrl
controllers/jsctrl
controllers/pyctrl
controllers/uppercase
controllers/guidance_ctrl
controllers/ag2_ctrl
aicirt
"

NATIVE="$(uname -s | tr 'A-Z' 'a-z')-$(uname -m)"

if test -z "$BUILD_TAG" ; then
    D=`date +%Y%m%d-%H%M`
    TAG=`git describe --dirty --tags --match 'v[0-9]*' --always | sed -e 's/^v//; s/-dirty/-'"$D/"`
else
    TAG="$BUILD_TAG"
fi

XZ=

if [ "$1" == "--xz" ] ; then
    XZ=1
    shift
fi

echo "Building for $NATIVE with tag $TAG..."

set -e

if test -z "$SKIP_LLAMA_CPP" ; then
    FOLDERS="$FOLDERS rllm/rllm-llamacpp"
fi

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

if [ "$SUFF" = "wasm32-wasi" ] ; then
    WASM=
    for B in $BN ; do
        case $B in
            aici_uppercase.wasm)
                # skip
                ;;
            aici_*.wasm)
                cp $T/$B target/dist/
                WASM="$WASM $(basename $B .wasm | sed -e 's/aici_//')"
                ;;
        esac
    done
    cat > $T/tag.sh <<EOF
#!/bin/bash
VERSION="$TAG"
MODULES="$(echo $WASM)"
EOF
    cat >> $T/tag.sh <<'EOF'
set -e
LATEST=0
PREFIX=
: ${AICI:=aici}
if [ "$1" == "--latest" ] ; then
    LATEST=1
    shift
fi
if [ "$1" != "" ] ; then
    PREFIX="$1"
    shift
fi
case "$PREFIX" in
    -*)
        echo "Usage: $0 [--latest] [prefix]"
        exit 1
        ;;
esac

echo "Tagging aici=$AICI prefix=$PREFIX latest=$LATEST version=$VERSION ..."
for M in $MODULES ; do
    TAG_PREF="$PREFIX$M"
    TAG="--tag $TAG_PREF-v$VERSION"
    if [ "$LATEST" = "1" ] ; then
        TAG="$TAG --tag $TAG_PREF-latest"
    fi
    echo "Tagging $M $TAG ..."
    $AICI upload aici_$M.wasm $TAG
done
EOF
    chmod +x $T/tag.sh
    BN="$BN tag.sh"
else
    (cd $T && strip $BN)
fi


echo >> $T/README.md
echo "Contents:" >> $T/README.md
echo '```' >> $T/README.md
(cd $T && ls -l $BN | awk '{print $5, $9}') >> $T/README.md
echo >> $T/README.md
(cd $T && sha256sum $BN) >> $T/README.md
echo '```' >> $T/README.md
echo >> $T/README.md

if [ "$SUFF" = "wasm32-wasi" ] ; then
    cat >> $T/README.md <<'EOF'
## Tagging

You can upload and tag the modules using the `tag.sh` script.

```
Usage: ./tag.sh [--latest] [prefix]
```

The `--latest` option will also tag the module with the `controller-latest` tag,
while the `prefix` will be prepended to the module name when tagging.

This requires `aici` command to be in path and AICI_API_BASE variable set.


EOF
fi

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
release aicirt "AICI Runtime" "$NATIVE" target/release/aicirt

if test -z "$SKIP_LLAMA_CPP" ; then
    release rllm-llamacpp "rLLM with llama.cpp" "$NATIVE" target/release/rllm-llamacpp
fi
