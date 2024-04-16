#!/bin/sh

set -e

if test -w target ; then
    :
else
    sudo chgrp docker target
    sudo chmod g+w target
fi

rm -f target/vllm.tar.gz
( cd py/vllm/vllm; \
  find . -name '*.py' -print0 | tar -czf - --null -T - ) > target/vllm.tar.gz
ls -l target/vllm.tar.gz

DOCKER_BUILDKIT=1 \
docker build . -f .devcontainer/Dockerfile-prod-vllm \
    --tag aici/vllm-openai
