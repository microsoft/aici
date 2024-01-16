#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python $HERE/../harness/bench_server.py "$@"
