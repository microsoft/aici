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

You can inspect test cases like so:

```
$ python scripts/testgen.py show expected/phi-1_5/lighthouse.safetensors 
Prompt: 'Write a detailed analogy between mathematics and a lighthouse.\n\nAnswer:'
Output: ' In mathematics, logic is like a beacon of the lighthouse. It guides us'
logits: torch.Size([15, 128]) min: tensor(12.7188) avg: tensor(17.6671) max: tensor(36.0938)
prob_mass: torch.Size([15]) min: tensor(0.9795) avg: tensor(0.9944) max: tensor(0.9999)
$ 
```

`prob_mass` refers to the sum of probiblites of the top 128 logits after softmax
(for every token of output). It should be very close to 1.

## Models

The following models have been tested:

* [CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf)
* [CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)
  - this barely fits in the 80GB A100, not much space for KV-cache
* [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
* [Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b)
* [phi-1_5](https://huggingface.co/microsoft/phi-1_5)
* [phi-2](https://huggingface.co/microsoft/phi-2)

In general all Llama models should work.

## Acknowledgements

See [top-level README.md](../README.md#acknowledgements).
