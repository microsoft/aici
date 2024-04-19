#!/bin/sh

P=`ps -ax|grep 'aicir[t]\|rllm-serve[r]\|/serve[r]\.sh\|node.*/buil[t]/worker|python[3] -m vllm.entrypoints' | awk '{print $1}' | xargs echo`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
  sleep 1
  kill -9 $P
fi
