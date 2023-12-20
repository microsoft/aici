#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python $HERE/../harness/serve_bench.py "$@"
