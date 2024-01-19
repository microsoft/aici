#!/bin/sh
tsc --version > /dev/null 2>&1 || npm install -g typescript
set -e
tsc -p ts
node gen-dts.mjs

