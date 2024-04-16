#!/bin/sh

PORT=4242
ADD_ARGS=
VLLM_ARGS="--port $PORT"
DOCKER_ARGS=""

case "$1" in
	--orca)
		shift
		ADD_ARGS="--model microsoft/Orca-2-13b --revision refs/pr/22 --aici-tokenizer=orca"
		;;
	--shell)
		shift
		DOCKER_ARGS="--entrypoint /bin/bash -it"
		VLLM_ARGS=""
		;;
esac

docker run \
		--privileged \
		--gpus=all \
		--shm-size=8g \
        --mount source=profile,target=/root,type=volume \
		--mount target=/root/.vscode-server,type=volume \
		-p $PORT:$PORT \
		$DOCKER_ARGS \
    aici/vllm-openai \
	$VLLM_ARGS \
	$ADD_ARGS \
	"$@"
