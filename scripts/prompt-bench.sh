#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python $HERE/../harness/prompt_bench.py "$@"