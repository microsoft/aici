#!/bin/sh

PORT=4242
ADD_ARGS=
VLLM_ARGS="--port $PORT"
DOCKER_ARGS=""

case "$1" in
	--phi2)
		shift
		ADD_ARGS="-m https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q8_0.gguf -t phi"
		;;
	--folder)
		shift
		D=`cd $1; pwd`
		DOCKER_ARGS="--mount type=bind,source=$D,target=/workspace/model"
		ADD_ARGS="-m ./model --aici-tokenizer ./model/tokenizer.json --tokenizer ./model"
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
        --mount source=profile,target=/root,type=volume \
		-p $PORT:$PORT \
		$DOCKER_ARGS \
    aici/rllm-llamacpp \
	$VLLM_ARGS \
	$ADD_ARGS \
	"$@"
