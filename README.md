# Artificial Intelligence Controller Interface (AICI)

The Artificial Intelligence Controller Interface (AICI)
lets you build Controllers that constrain and direct output of a Large Language Model (LLM) in real time.
Controllers are light-weight [WebAssembly](https://webassembly.org/) (Wasm) modules
which run in on the same machine as the LLM inference engine, utilizing the CPU while the GPU is busy
with token generation. AICI is:

- **secure**: Wasm modules are [sandboxed](#security) and cannot access the filesystem, network, or any other resources
- **fast**: Wasm modules are compiled to native code and run in parallel with the LLM inference engine, inducing only a minimal overhead to the generation process
- **flexible**: Wasm modules can be generated in any language that can compile to Wasm

This repository contains:

- [definition](aici_abi/README.md#low-level-interface) of the AICI binary interface
- [aici_abi](aici_abi) - a Rust crate for easily implementing controllers (Wasm modules adhering to AICI)
- [aicirt](aicirt) - an implementation of a runtime for controllers,
  built on top [Wasmtime](https://wasmtime.dev/);
  LLM inference engines talk to aicirt via shared memory and semaphores
- [rLLM](rllm) - a reference implementation of an LLM inference engine
- [pyaici](pyaici) - a Python package for interacting with aicirt and running controllers;
  includes `aici` command-line tool
- [promptlib](promptlib) - a Python package that exposes API for easily creating and running DeclCtrl ASTs
  (will change to generate PyCtrl programs in the future)

And a number of sample/reference controllers:

- [uppercase](uppercase) - a sample/starter project for aici_abi
- [PyCtrl](pyctrl) - an embedded Python 3 interpreter (using [RustPython](https://github.com/RustPython/RustPython)),
  which lets you write controllers in Python
- [JsCtrl](jsctrl) - an embedded JavaScript interpreter (using [QuickJS](https://bellard.org/quickjs/)),
  which lets you write controllers in JavaScript
- [DeclCtrl](declctrl) - a controller that interprets a simple JSON AST (Abstract Syntax Tree) to specify constraints

Everything above implemented in Rust, unless otherwise stated.

AICI abstract LLM inference engine from the controller and vice-versa, as in the picture below.
The rounded nodes are asiprational.
Additional layers can be built on top - we provide [promptlib](promptlib),
but we strongly believe [Guidance](https://github.com/guidance-ai/guidance) and
[LMQL](https://lmql.ai/) can also run on top of AICI (either with custom controllers or utilizing PyCtrl or JsCtrl).

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
- if you want to run the inference server (rllm) locally, use the **AICI with CUDA** container;
  this requires a CUDA-capable GPU (currently only 8.0 (A100) is supported)
- finally, if you want to try the AICI integration with vLLM, use the
  **AICI with CUDA and vLLM (experimental)** container

Each of the above containers takes longer than the previous one to build.

If you're not familiar with [devcontainers](https://containers.dev/),
you need to install the [Dev Containers VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
and from the command palette in VSCode select **Dev Containers: Reopen in Container...**.
It pops a list of available devcontainers, select the one you want to use.

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
You can also try other models, see [rllm/README.md](rllm/README.md) for details.

## Security

- `aicirt` runs in a separate process, and can run under a different user than the LLM engine
- Wasm modules are [sandboxed by Wasmtime](https://docs.wasmtime.dev/security.html)
- Wasm only have access to [`aici_host_*` functions](aici_abi/src/host.rs),
  implemented in [hostimpl.rs](aicirt/src/hostimpl.rs)
- `aicirt` also exposes a partial WASI interface; however almost all the functions are no-op, except
  for `fd_write` which shims file descriptors 1 and 2 (stdout and stderr) to print debug messages

In particular, Wasm modules cannot access the filesystem, network, or any other resources.
They also cannot spin threads or access any timers (this is relevant for Spectre/Meltdown attacks).

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
