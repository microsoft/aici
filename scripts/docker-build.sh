#!/bin/sh

set -e

if test -w target ; then
    :
else
    sudo chgrp docker target
    sudo chmod g+w target
fi

if test -f target/dist/README.md ; then
    :
else
    echo "Run ./scripts/release.sh first (in dev docker container)"
    exit 1
fi

set -x
rm -f target/vllm.tar.gz
( cd py/vllm/vllm; \
  find . -name '*.py' -print0 | tar -czf - --null -T - ) > target/vllm.tar.gz
ls -l target/vllm.tar.gz

DOCKER_BUILDKIT=1 \
docker build . -f .devcontainer/Dockerfile-prod-vllm \
    --tag aici/vllm-openai

if [ "X$DOCKER_PUSH" != X ] ; then
    if test -z "$DOCKER_TAG" ; then
        DOCKER_TAG=v$(date '+%Y%m%d-%H%M')
    fi
    docker tag aici/vllm-openai $DOCKER_PUSH:$DOCKER_TAG
    docker push $DOCKER_PUSH:$DOCKER_TAG

    set +x
    echo
    echo "Pushed $DOCKER_PUSH:$DOCKER_TAG"
    echo
fi
