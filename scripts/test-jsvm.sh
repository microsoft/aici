#!/bin/sh

set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../jsvm
tsc -p ts
PYTHONPATH=$HERE/.. \
python ../pyvm/driver.py ts/sample.js
