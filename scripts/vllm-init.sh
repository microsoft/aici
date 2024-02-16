#!/bin/sh

set -x
set -e
mkdir -p tmp
if test -f py/vllm/setup.py; then
    :
else
    git submodule update --init --recursive
fi
cd py/vllm
python setup.py develop
