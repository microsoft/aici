# rLLM for llama.cpp

This is similar to the [CUDA-based rLLM](../rllm/)
but built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Building

If you're not using the supplied docker container follow the
[build setup instructions](../README.md#build-setup-on-linux-including-wsl2).

To compile and run first aicirt and then the rllm server, run:

```bash
./cpp-server.sh phi2
```

You can also try passing `--cuda` before `phi2`.
