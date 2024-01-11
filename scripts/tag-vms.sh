#!/bin/sh

TIMESTAMP=`date --utc '+%+4Y-%m-%d-%H%M'`

VMS="$*"
if [ X"$VMS" = X ]; then
    VMS="declctrl pyctrl jsctrl"
fi

for vm in $VMS ; do
    ./scripts/aici.sh build $vm -T $vm-latest -T $vm-$TIMESTAMP
done
