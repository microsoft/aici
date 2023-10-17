#!/bin/sh

set -x
set -e
git submodule update --init --recursive
cd vllm
pip install --verbose -e .