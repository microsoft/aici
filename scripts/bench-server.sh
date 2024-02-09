#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/../py \
python3 $HERE/py/bench_server.py "$@"
