# rLLM

This is a partial port of [vLLM](https://github.com/vllm-project/vllm)
to Rust and [tch-rs](https://github.com/LaurentMazare/tch-rs).
It is mostly meant as a proving ground for AICI (AI Controller Interface) integration.


## Building

If you're not using the supplied docker container make sure to check
that the following environment variables are set:

```bash
export CUDA_COMPUTE_CAP="80"
export LIBTORCH_USE_PYTORCH="1"
```

You can run the server with `./server.sh` script; have a look inside to figure out
how to run with different options.

## Tests

The `expected/` directory contains sample prompts along with expected model output -
top 128 logits for the first few tokens of output.
Running `./expected/tests.sh` will run rLLM on these testcases and make sure it gets the
same logits with some tolerance.

## Credits

Some OpenAI JSON datatype definitions are copied from
[candle-vllm](https://github.com/EricLBuehler/candle-vllm/tree/master/src/openai).

Parts of code are direct port of Python from [vLLM](https://github.com/vllm-project/vllm).

Bits of model implementations are influenced by
[candle-transformers](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models).

