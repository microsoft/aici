#!/bin/sh

cd "`dirname $0`/.."

TIMESTAMP=`date --utc '+%+4Y-%m-%d-%H%M'`

CTRLS="$*"
if [ X"$CTRLS" = X ]; then
    CTRLS="declctrl pyctrl jsctrl llguidance_ctrl"
fi

for ctrl in $CTRLS ; do
    bctrl=$(echo $ctrl | sed -e 's/_ctrl//')
    ./aici.sh build controllers/$ctrl -T $ctrl-latest -T $ctrl-$TIMESTAMP -T $bctrl
done
