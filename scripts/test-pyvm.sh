#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../pyvm
PYTHONPATH=$HERE/.. \
python driver.py samples/test*.py
