#!/bin/sh

F="
aici_abi
aici_ast_runner
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
