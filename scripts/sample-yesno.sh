#!/bin/sh
PROMPT="$*"
if [ -z "$PROMPT" ]; then
    PROMPT="Is coffee any good?"
fi
set -x
./scripts/aici.sh run --build aici_abi::yesno --prompt "$PROMPT"
