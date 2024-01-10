#!/bin/sh

TIMESTAMP=`date --utc '+%+4Y-%m-%d-%H%M'`

for vm in declvm pyvm jsvm ; do
    ./scripts/aici.sh build $vm -T $vm-latest -T $vm-$TIMESTAMP
done
