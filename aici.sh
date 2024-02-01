#!/bin/sh

if [ "X$AICI_API_BASE" = "X" ] ; then
  export AICI_API_BASE="http://127.0.0.1:4242/v1/"
fi

PYTHONPATH=`dirname $0` \
python3 -m pyaici.cli "$@"
