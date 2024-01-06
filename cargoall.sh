#!/bin/sh

F="
aici_abi
declvm
aicirt
pyvm
rllm
"

set -e
set -x
for f in $F ; do
    cd $f
    cargo "$@"
    cd ..
done
