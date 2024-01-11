#!/bin/sh

TIMESTAMP=`date --utc '+%+4Y-%m-%d-%H%M'`

CTRLS="$*"
if [ X"$CTRLS" = X ]; then
    CTRLS="declctrl pyctrl jsctrl"
fi

for ctrl in $CTRLS ; do
    ./aici.sh build $ctrl -T $ctrl-latest -T $ctrl-$TIMESTAMP
done
