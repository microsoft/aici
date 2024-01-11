#!/bin/sh
PROMPT="$*"
if [ -z "$PROMPT" ]; then
    PROMPT="Is coffee any good?"
fi
set -x
./aici.sh run --build aici_abi::yesno --prompt "$PROMPT"
