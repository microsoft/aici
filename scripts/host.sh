#!/bin/bash

#
# This is for running an inference server, including a tunnel.
#

set -e

CONT=rllm
INNER=0
INNER_STOP=0
FULL=0
STOP=0
START_CONTAINER=0
PULL=1
CPP=0

WS=`cd $(dirname $0)/..; pwd`

test -f .devcontainer/Dockerfile-cuda || exit 1

while [ $# -gt 0 ] ; do
    case "$1" in
        --no-pull ) PULL=0 ;;
        --env )
            . "$2"
            FOLDER=`dirname $2`
            shift
            ;;
        --in-screen ) INNER=screen ;;
        --start-tunnel ) INNER=tunnel ;;
        --start-model ) INNER=model ;;
        --full ) FULL=1 ;;
        --stop ) STOP=1 ;;
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

if [ "$INNER" = "screen" ] ; then
    docker_cmd "cd tmp/ws-http-tunnel && source /usr/local/nvm/nvm.sh && yarn compile-client"
    for f in tmp/models/*/.env ; do
        . $f
        screen "$0" --start-tunnel --env $f
        screen "$0" --start-model  --env $f
    done
    sleep 3
    exit 0
fi

if [ "$INNER" = "tunnel" ] ; then
    echo "in tunnel for $MODEL in $FOLDER"
    WORKER="/workspaces/aici/tmp/ws-http-tunnel/built/worker.js"
    docker_cmd "cd $FOLDER && source /usr/local/nvm/nvm.sh && while : ; do node $WORKER ; sleep 2 ; done"
    exit 0
fi

if [ "$INNER" = "model" ] ; then
    echo "in server for $MODEL in $FOLDER"
    PREF="cd $FOLDER && CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    ARGS="--port $FWD_PORT --shm-prefix /aici-${MODEL}-"
    if [ "$CPP" -eq 1 ] ; then
        docker_cmd "$PREF /workspaces/aici/rllm-cpp/cpp-server.sh --loop --cuda $MODEL $ARGS"
    else
        docker_cmd "$PREF /workspaces/aici/rllm-cuda/server.sh --loop $MODEL $ARGS"
    fi
    exit 0
fi

if [ "$PULL" = 1 ] ; then
    echo "Pulling..."
    git pull
    (cd tmp/ws-http-tunnel && git pull)
fi

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
# P=`ps -ax|grep 'docker [e]xec' | awk '{print $1}' | xargs echo`
# if [ "X$P" != "X" ] ; then 
#   echo "KILL $P"
#   kill $P
# fi

docker_cmd "./scripts/kill-server.sh"

echo "Building ..."
docker_cmd "cd rllm-cuda && ./server.sh build"

screen -wipe >/dev/null || :

if [ $STOP = 1 ] ; then
    echo "Stopped."
    exit 0
fi

echo "Starting screen..."
screen "$0" --in-screen
