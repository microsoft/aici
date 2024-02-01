#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python3 $HERE/../harness/bench_server.py "$@"
