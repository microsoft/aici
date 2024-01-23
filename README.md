# Artificial Intelligence Controller Interface (AICI)

The Artificial Intelligence Controller Interface (AICI)
lets you build Controllers that constrain and direct output of a Large Language Model (LLM) in real time.
Controllers are light-weight WebAssembly (Wasm) modules
which run in on the same machine as the LLM inference engine, utilizing the CPU while the GPU is busy
with token generation.

AICI is a prototype, designed and built at [Microsoft Research](https://www.microsoft.com/en-us/research/).

AICI is:

- [Secure](#security): Controllers are sandboxed and cannot access the filesystem, network, or any other resources
- [Fast](#performance): Wasm modules are compiled to native code and run in parallel with the LLM inference engine, inducing only a
  minimal overhead to the generation process
- [Flexible](#flexibility): Controllers can be written in any language that can compile to Wasm (Rust, C, C++, ...),
  or be interpreted inside Wasm (Python, JavaScript, ...)

This repository contains:

- [definition](aici_abi/README.md#low-level-interface) of the AICI binary interface
- [aici_abi](aici_abi) - a Rust crate for easily implementing controllers (Wasm modules adhering to AICI)
- [aicirt](aicirt) - an implementation of a runtime for running controllers,
  built on top Wasmtime;
  LLM inference engines talk to aicirt via shared memory and semaphores
- [rLLM](rllm) - a reference implementation of an LLM inference engine built on libtorch, inspired by vLLM
- [rLLM-llama-cpp](cpp-rllm) - rLLM running on top of llama.cpp instead of libtorch
- [pyaici](pyaici) - a Python package for interacting with aicirt and running controllers;
  includes `aici` command-line tool
- [promptlib](promptlib) - a Python package that exposes API for easily creating and running DeclCtrl ASTs
  (will change to generate PyCtrl programs in the future)

And a number of sample/reference controllers:

- [uppercase](uppercase) - a sample/starter project for aici_abi
- [PyCtrl](pyctrl) - an embedded Python 3 interpreter (using RustPython),
  which lets you write controllers in Python
- [JsCtrl](jsctrl) - an embedded JavaScript interpreter (using QuickJS),
  which lets you write controllers in JavaScript
- [DeclCtrl](declctrl) - a controller that interprets a simple JSON AST (Abstract Syntax Tree) to specify constraints

Everything above implemented in Rust, unless otherwise stated,
and all controllers compile to [Wasm](https://webassembly.org/).

AICI abstract LLM inference engine from the controller and vice-versa, as in the picture below.
The rounded nodes are aspirational.
Additional layers can be built on top - we provide [promptlib](promptlib),
but we strongly believe that
[Guidance](https://github.com/guidance-ai/guidance),
[LMQL](https://lmql.ai/),
[Outlines](https://github.com/outlines-dev/outlines),
[jsonformer](https://github.com/1rgs/jsonformer),
[LMFE](https://github.com/noamgat/lm-format-enforcer),
etc.
can also run on top of AICI (either with custom controllers or utilizing PyCtrl or JsCtrl).

```mermaid
graph TD
    PyCtrl -- AICI --> aicirt[AICI-runtime]
    JsCtrl -- AICI --> aicirt
    guidance([GuidanceCtrl]) -- AICI --> aicirt
    lmql([LMQL Ctrl]) -- AICI --> aicirt
    aicirt -- POSIX SHM --> rLLM
    aicirt -- POSIX SHM --> llama([llama.cpp])
    aicirt -- POSIX SHM --> pyaici
    pyaici -- Python --> vLLM(vLLM)
    pyaici -- Python --> hf(HF Transformers)
```

The [pyaici](pyaici) package makes it easier to integrate AICI with Python-based LLM inference engines.
The support for [HuggingFace Transformers](harness/run_hf.py)
and [vLLM REST server](harness/vllm_server.py) is currently out of date
and llama.cpp is in plans.
Please use the [rLLM](rllm) for now.

## Getting started

There are several levels at which you can use AICI.

- you can use the provided PyCtrl, JsCtrl or DeclCtrl on a remote server;
  no devcontainer is required in that case; [more info](proxy.md)
- you can modify one of the provided controllers or build a new one;
  this typically requires rust, and the preferred way to work with it is to use the
  provided **AICI Client-side** devcontainer - it should work on any machine with Docker and VSCode
- you can also build the [rLLM-llama-cpp](cpp-rllm) and run it locally;
  the same **AICI Client-side** devcontainer should work
- if you want to run the inference server (rllm) locally, use the **AICI with CUDA** container;
  this requires a CUDA-capable GPU (currently only 8.0 (A100) is supported)
- finally, if you want to try the AICI integration with vLLM, use the
  **AICI with CUDA and vLLM (experimental)** container

Each of the above containers takes longer than the previous one to build.

If you're not familiar with [devcontainers](https://containers.dev/),
you need to install the [Dev Containers VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
and from the command palette in VSCode select **Dev Containers: Reopen in Container...**.
It pops a list of available devcontainers, select the one you want to use.

### Build setup on Linux (including WSL2)

This should be roughly equivalent to the **AICI Client-side** devcontainer.
See also [common.dockerfile](.devcontainer/common.dockerfile).

- install required packages; it's likely you already have some or all of these
  but the list should be exhaustive for fresh Ubuntu-22.04 install in WSL

```bash
sudo apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache \
    cmake curl libjpeg-dev libpng-dev \
    strace linux-tools-common linux-tools-generic \
    llvm-dev libclang-dev clang ccache apache2-utils git-lfs \
    screen bsdmainutils pip python3-dev python-is-python3 \
    nodejs npm pkg-config

pip install pytest pytest-forked ujson posix_ipc numpy requests
```

- [install](https://www.rust-lang.org/tools/install) rustup and restart current shell

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

- install rustup components:

```bash
rustup target add wasm32-wasi
rustup component add rustfmt
```

### Interacting with server

To get started interacting with a cloud AICI server first export the API key.
If running local server, leave `AICI_API_BASE` unset.

```bash
export AICI_API_BASE="https://inference.example.com/v1/#key=wht_..."
```

Now, use query the model with or without AICI Controller:

```bash
./aici.sh infer --prompt "The answer to the ultimate question of life"
./aici.sh run pyctrl/samples/test.py
./aici.sh run declctrl/arg2.json
./aici.sh run --build aici_abi::uppercase
```

Run `./aici.sh -h` to see usage info.

If the server is running with Orca-2 13B model,
you can also run tests with `pytest` for the DeclCtrl, or with `./scripts/test-pyctrl.sh` for PyCtrl.

### Running local server

To run rLLM server, go to `rllm/` and run `./server.sh orca`.
This will run the inference server with Orca-2 13B model (which is expected by testcases).
If you don't have CUDA, go to `cpp-rllm/` and run `./cpp-server.sh phi2`.
You can also try other models, see [rllm/README.md](rllm/README.md) and
[cpp-rllm/README.md](cpp-rllm/README.md) for details.

## Security

- `aicirt` runs in a separate process, and can run under a different user than the LLM engine
- Wasm modules are [sandboxed by Wasmtime](https://docs.wasmtime.dev/security.html)
- Wasm only have access to [`aici_host_*` functions](aici_abi/src/host.rs),
  implemented in [hostimpl.rs](aicirt/src/hostimpl.rs)
- `aicirt` also exposes a partial WASI interface; however almost all the functions are no-op, except
  for `fd_write` which shims file descriptors 1 and 2 (stdout and stderr) to print debug messages
- each Wasm module runs in a separate process, helping with Spectre/Meltdown mitigation
  and allowing limits on CPU usage

In particular, Wasm modules cannot access the filesystem, network, or any other resources.
They also cannot spin threads or access any timers (this is relevant for Spectre/Meltdown attacks).

## Performance

Most of computation in AICI Controllers occurs on the CPU, in parallel with the logit generation on the GPU.
The generation occurs in steps, where logits are generated in parallel for a new token for each sequence in a batch
(typically between 1 and 50).
This involves reading the whole model and KV caches for sequences in the batch from the GPU memory.
For optimal batch throughput, the model and KV caches should utilize a major fraction of the GPU memory,
and reading the whole memory takes about 40ms on A100 GPU (80GB).

Thus, each step of generation takes on the order of 20-50ms.
With careful engineering,
this is more than enough to compute the set of allowed tokens in Rust compiled to Wasm.
These can be combined either natively in Rust, or via Python or JavaScript interpreters
we provide.

For example, computing allowed token set in the 32000-strong vocabulary of Llama model takes:

- about 2.0ms for Yacc grammar of the C programming language
- about 0.3ms for a regular expression
- about 0.2ms for a substring contraint, from 4kB string

The above numbers are for a single sequence, however each sequence is processed in separate process,
and thus if there is more cores than sequences (which is typical), they do not change.
They also include overhead of calling into Python interpreter implemented in Wasm, and then back into
Rust-generated Wasm code for the constraint itself.
They are all well within the 20-50ms budget, so do not affect the generation time at all.

There is also some overhead in the critical path of sampling. It comes down to about 0.3ms per generation step
when executing 10 sequences in parallel (this is irrespective of the constraint used).
The overhead goes up to around 0.7ms for 40 sequences (though it has not been fully optimized yet).

WebAssembly is designed to have minimal overhead, compared to native code.
In our experience, [highly optimized](aici_abi/implementation.md#token-trie)
Rust code is less than 2x slower when run in
[Wasmtime](https://wasmtime.dev/) than native.
This is 10-100x better than JavaScript or Python.

All measurements done on AMD EPYC 7V13 with nVidia A100 GPU with 80GB of VRAM.

## Flexibility

The low-level interface that AICI runtime provides allows for:

- interaction with the LLM inference engine before, during, and after every generated token
- constraining decoding to a set of tokens
- backtracking KV-cache to a previous state
- fast-forwarding several tokens at a time (if they are known)
- forking generation into multiple branches
- communication between forks via shared variables
- utility functions for converting between tokens and byte strings

It can be utilized from any language that compiles to Wasm.

This repository provides a Rust library that makes it easy to implement controllers in Rust,
and provides [efficient implementations](aici_abi/implementation.md)
of specific constraints ([regular expressions](aici_abi/README.md#regular-expressions),
[yacc grammars](aici_abi/README.md#lr1-grammars), substrings).
We also provide [Python](pyctrl) and [JavaScript](jsctrl) interpreters
that allow to glue these constraints together.
All of these can be easily extended.

## Acknowledgements

- [Flash Attention kernels](tch-cuda/kernels/flash_attn/) are copied from
  [flash-attention repo](https://github.com/Dao-AILab/flash-attention);
  see [BSD LICENSE](tch-cuda/kernels/flash_attn/LICENSE)
- [Paged Attention kernels](tch-cuda/kernels/vllm/) are copied from
  [vLLM repo](https://github.com/vllm-project/vllm);
  see [Apache LICENSE](tch-cuda/kernels/vllm/LICENSE)
- [OpenAI API definitions](rllm/src/server/openai/) are copied and modified from
  [candle-vllm](https://github.com/EricLBuehler/candle-vllm);
  see [MIT LICENSE](rllm/src/server/openai/LICENSE)
- [cache_engine.rs](rllm/src/paged/cache_engine.rs),
  [config.rs](rllm/src/config.rs),
  and [scheduler.rs](rllm/src/paged/scheduler.rs)
  are loosely based on [vLLM](https://github.com/vllm-project/vllm)
- [llama.rs](rllm/src/llm/llama.rs), [phi.rs](rllm/src/llm/phi.rs)
  and [logits.rs](rllm/src/logits.rs) are based on
  [candle-transformers](https://github.com/huggingface/candle/tree/main/candle-transformers)
- the [example ANSI C grammar](aici_abi/grammars/c.y) is based on
  https://www.lysator.liu.se/c/ANSI-C-grammar-y.html by Jeff Lee (from 1985)

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
