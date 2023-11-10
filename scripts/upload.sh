#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python $HERE/upload.py "$@"