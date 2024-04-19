#!/bin/bash

N=10
if [ -n "$1" ]; then
    N=$1
fi

mkdir -p tmp
rm -f tmp/fail

for n in $(seq $N) ; do
    echo "Start $n"
    if ./scripts/test-pyctrl.sh > tmp/logs-$n.txt 2>&1 ; then
        echo "Passed test $n"
    else
        echo "Failed test $n; see tmp/logs-$n.txt"
        echo $n >> tmp/fail
    fi &
    sleep 1
done

wait

if [ -f tmp/fail ]; then
    echo "Some tests failed; see tmp/fail"
    exit 1
fi
