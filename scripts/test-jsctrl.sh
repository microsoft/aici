#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../jsctrl
tsc -p samples
PYTHONPATH=$HERE/.. \
python ../pyctrl/driver.py samples/dist/test.js

