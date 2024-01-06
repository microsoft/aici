#!/bin/sh

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python -m pyaici.cli "$@"