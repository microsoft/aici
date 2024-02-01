#!/bin/sh

set -e

./expected/go.sh \
expected/phi-1_5 \
expected/orca

if [ "$1" = "all" ] ; then
./expected/go.sh \
expected/codellama34 \
expected/codellama \
expected/phi-2 \
expected/llama
fi

