#!/bin/bash

set -e

MODEL=orca
CONT=rllm
INNER=0
INNER_STOP=0
FULL=0
START_CONTAINER=0

WS=`cd $(dirname $0)/..; pwd`

test -f .devcontainer/Dockerfile-cuda || exit 1

while [ $# -gt 0 ] ; do
    case "$1" in
        --model )
            MODEL="$2"
            shift
            ;;
        --inner1 ) INNER=1 ;;
        --inner2 ) INNER=2 ;;
        --full ) FULL=1 ;;
        * )
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

function docker_cmd() {
    docker exec -w /workspaces/aici -it $CONT /bin/sh -c "$*"
}

if [ "$INNER" = 1 ] ; then
    echo "Running inner..."
    docker_cmd "cd rllm && ./server.sh loop $MODEL"
    exit 0
fi

if [ "$INNER" = 2 ] ; then
    echo "Running inner2..."
    screen "$0" --inner1
    docker_cmd "cd tmp/ws-http-tunnel && source /usr/local/nvm/nvm.sh && yarn worker"
    exit 0
fi

echo "Pulling..."
git pull
(cd tmp/ws-http-tunnel && git pull)

if [ "$FULL" = 1 ] ; then
    echo "Building full..."
    docker build -f .devcontainer/Dockerfile-cuda .devcontainer -t rllm-server:latest
    START_CONTAINER=1
fi

if [ $START_CONTAINER = 0 ] && docker_cmd "true" ; then
    echo "Container already running"
else
    echo "Container not found, will start"
    START_CONTAINER=1
fi

if [ $START_CONTAINER = 1 ] ; then
    echo "Cleaning up containers..."
    docker stop -t 2 $CONT || :
    docker rm $CONT || :

    echo "Running new container..."
    docker run --sig-proxy=false \
        --mount type=bind,source=$WS,target=/workspaces/aici \
        --mount source=profile,target=/root,type=volume \
        --mount source=cargo-git,target=/usr/local/cargo/git,type=volume \
        --mount source=cargo-registry,target=/usr/local/cargo/registry,type=volume \
        --privileged --gpus all --shm-size=8g \
        --name $CONT -d \
        rllm-server:latest /bin/sh -c "while : ; do sleep 100 ; done"

    # do some magical nvidia setup; without it the first run of server is super-slow
    docker exec -it $CONT /opt/nvidia/nvidia_entrypoint.sh nvidia-smi
fi

echo "Stopping inner servers..."
docker_cmd "./scripts/kill-server.sh"

echo "Building ..."
docker_cmd "cd rllm && ./server.sh build"

screen -wipe >/dev/null || :

echo "Starting screen..."
screen "$0" --inner2
