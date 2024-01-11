#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../jsctrl
tsc -p ts
PYTHONPATH=$HERE/.. \
python ../pyctrl/driver.py ts/sample.js
