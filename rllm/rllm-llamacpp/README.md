# rLLM for llama.cpp

This is similar to the [CUDA-based rLLM](../rllm-cuda/)
but built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Building

If you're not using the supplied docker container follow the
[build setup instructions](../../README.md#development-environment-setup).

To compile and run first aicirt and then the rllm server, run:

```bash
./server.sh phi2
```

You can also try passing `--cuda` before `phi2`.
