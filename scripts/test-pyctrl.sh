#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../pyctrl
PYTHONPATH=$HERE/.. \
python3 driver.py samples/test*.py
