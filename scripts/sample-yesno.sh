#!/bin/sh
PROMPT="$*"
if [ -z "$PROMPT" ]; then
    PROMPT="Is coffee any good?"
fi
set -x
echo "$PROMPT" | ./aici.sh run --build aici_abi::yesno -
