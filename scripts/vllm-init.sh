#!/bin/sh

set -x
set -e
mkdir -p tmp
if test -f vllm/setup.py; then
    :
else
    git submodule update --init --recursive
fi
cd vllm
pip install --verbose -e .
