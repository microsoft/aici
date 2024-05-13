#!/bin/sh

set -e

DOCKER_TARGET="$1"
if test -z "$DOCKER_TARGET" ; then
    DOCKER_TARGET=vllm-general
fi

DOCKERFILE="$2"
if test -z "$DOCKERFILE" ; then
    DOCKERFILE=.devcontainer/Dockerfile-prod-vllm
fi

D=`date +%Y%m%d-%H%M`
TAG=`git describe --dirty --tags --match 'v[0-9]*' --always | sed -e 's/^v//; s/-dirty/-'"$D/"`

set -x

DOCKER_BUILDKIT=1 \
docker build . -f $DOCKERFILE \
    --target $DOCKER_TARGET \
    --tag $DOCKER_TARGET \
    --build-arg tag="$TAG" \
    --progress=plain

if [ "X$DOCKER_PUSH" != X ] ; then
    if test -z "$DOCKER_TAG" ; then
        DOCKER_TAG=v$(date '+%Y%m%d-%H%M')
    fi
    docker tag $DOCKER_TARGET $DOCKER_PUSH:$DOCKER_TAG
    docker push $DOCKER_PUSH:$DOCKER_TAG

    set +x
    echo
    echo "Pushed $DOCKER_PUSH:$DOCKER_TAG"
    echo
fi
