#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../controllers/pyctrl
PYTHONPATH=$HERE/../py \
python3 driver.py samples/test*.py "$@"
