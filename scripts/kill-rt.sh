#!/bin/sh

P=`ps -ax|grep 'aicir[t]' | awk '{print $1}'`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi
