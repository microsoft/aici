# candle-vllm
[![Continuous integration](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml/badge.svg)](https://github.com/EricLBuehler/candle-vllm/actions/workflows/ci.yml)

Efficient platform for inference and serving local LLMs including an OpenAI compatible API server.

## Features
- OpenAI compatible API server provided for serving LLMs.
- Highly extensible trait-based system to allow rapid implementation of new module pipelines,
- Streaming support in generation.

## Pipelines
- Llama
- Mistral

## Overview
One of the goals of `candle-vllm` is to interface locally served LLMs using an OpenAI compatible API server.

1) During initial setup: the model, tokenizer and other parameters are loaded.

2) When a request is received:
  1) Sampling parameters are extracted, including `n` - the number of choices to generate.
  2) The request is converted to a prompt which is sent to the model pipeline.
      - If a streaming request is received, token-by-token streaming using SSEs is established (`n` choices of 1 token).
      - Otherwise, a `n` choices are generated and returned.

## Contributing
The following features are planned to be implemented, but contributions are especially welcome:
- Sampling methods:
  - Beam search ([huggingface/candle#1319](https://github.com/huggingface/candle/issues/1319))
  - `presence_penalty` and `frequency_penalty`
- Pipeline batching ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- KV cache ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- PagedAttention ([#3](https://github.com/EricLBuehler/candle-vllm/issues/3))
- More pipelines (from `candle-transformers`)

## Resources
- Python implementation: [`vllm-project`](https://github.com/vllm-project/vllm)
- [`vllm` paper](https://arxiv.org/abs/2309.06180)
