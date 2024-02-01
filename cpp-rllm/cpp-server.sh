#!/bin/sh

CPP=1 exec `dirname $0`/../rllm-cuda/server.sh "$@"
