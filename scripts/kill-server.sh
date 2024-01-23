#!/bin/sh

P=`ps -ax|grep 'aicir[t]\|rllm-serve[r]\|\./server\.sh\|node \./built/worker' | awk '{print $1}' | xargs echo`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi
