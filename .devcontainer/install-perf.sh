#!/usr/bin/env bash

set -euo pipefail

UNAME_R=$(uname -r)
if [[ $UNAME_R == *"-microsoft-standard-WSL2" ]]; then
  apt-get install -y linux-tools-generic
else
  apt-get install -y "linux-tools-$(uname -r)"
fi
