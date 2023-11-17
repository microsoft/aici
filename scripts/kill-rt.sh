#!/bin/sh

P=`ps fax|grep 'aicir[t]' | awk '{print $1}'`

if [ "X$P" != "X" ] ; then 
  echo "KILL $P"
  kill $P
fi
