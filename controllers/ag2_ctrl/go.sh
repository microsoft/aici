#!/bin/sh

F="$1"
if [ -z "$F" ] ; then
  F=run_g.py
fi

set -x
cd `dirname $0`
HERE=`pwd`
PYTHONPATH=$HERE/../../py:$HERE/../../py/guidance \
python3 $HERE/$F

