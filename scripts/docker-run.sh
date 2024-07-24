#!/bin/sh

PORT=4242
ADD_ARGS=
VLLM_ARGS="--port $PORT"
DOCKER_ARGS=""

if test -z "$DOCKER_TARGET" ; then
    DOCKER_TARGET=vllm-general
fi


case "$1" in
    --orca)
        shift
        ADD_ARGS="--model microsoft/Orca-2-13b --revision refs/pr/22 --aici-tokenizer=orca"
        ;;
    --folder)
        shift
        D=`cd $1; pwd`
        DOCKER_ARGS="--mount type=bind,source=$D,target=/vllm-workspace/model"
        ADD_ARGS="--model ./model --aici-tokenizer ./model/tokenizer.json --tokenizer ./model"
        shift
        ;;
    --shell)
        shift
        DOCKER_ARGS="--entrypoint /bin/bash -it"
        VLLM_ARGS=""
        ;;
esac

set -x
docker run \
        --privileged \
        --gpus=all \
        --shm-size=8g \
        --mount source=profile,target=/root,type=volume \
        -p $PORT:$PORT \
        $DOCKER_ARGS \
    $DOCKER_TARGET \
    $VLLM_ARGS \
    $ADD_ARGS \
    "$@"
