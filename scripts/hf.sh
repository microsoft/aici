#!/bin/sh

set -e
set -x

RUST_LOG=info,tokenizers=error,rllm=debug,aicirt=info \
PYTHONPATH=py \
python3 scripts/py/run_hf.py \
    --aici-rt ./target/release/aicirt \
    --controller gh:microsoft/aici/pyctrl \
    --controller-arg controllers/pyctrl/samples/phi.py  \
    --aici-tokenizer phi \
    --model microsoft/phi-2
