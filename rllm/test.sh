#!/bin/sh

set -e

./expected/go.sh \
expected/phi-1_5 \
expected/phi-2 \
expected/codellama \
expected/orca

if [ "$1" = "all" ] ; then
./expected/go.sh \
expected/codellama34 \
expected/llama
fi

