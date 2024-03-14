#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
PYTHONPATH=$HERE/../../py:$HERE/../../py/guidance \
python3 $HERE/run_g.py

