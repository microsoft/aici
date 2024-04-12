#!/bin/sh

set -x
set -e
./scripts/test-pyctrl.sh
./scripts/test-jsctrl.sh
pytest
