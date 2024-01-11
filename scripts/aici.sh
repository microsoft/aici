#!/bin/sh

if [ "X$AICI_API_BASE" = "X" ] ; then
  export AICI_API_BASE="http://127.0.0.1:8080/v1/"
fi

HERE=`dirname $0`
PYTHONPATH=$HERE/.. \
python -m pyaici.cli "$@"