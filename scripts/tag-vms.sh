#!/bin/sh

TIMESTAMP=`date --utc '+%+4Y-%m-%d-%H%M'`

./scripts/aici.sh build declvm -T declvm-latest -T declvm-$TIMESTAMP
./scripts/aici.sh build pyvm -T pyvm-latest -T pyvm-$TIMESTAMP
