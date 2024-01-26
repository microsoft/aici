#!/bin/sh

P=`ps -ax|grep 'aicir[t]\|rllm-serve[r]\|/serve[r]\.sh\|node.*/buil[t]/worker' | awk '{print $1}' | xargs echo`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi
