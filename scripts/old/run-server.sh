#!/bin/sh

python -m vllm.entrypoints.api_server --port 8080 --host 127.0.0.1
