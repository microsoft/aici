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

- the [definition](aici_abi/README.md#low-level-interface) of the AICI binary interface
- [aici_abi](aici_abi) - a Rust crate for easily implementing controllers (Wasm modules adhering to AICI)
- [aicirt](aicirt) - an implementation of a runtime for controllers,
  built on top [Wasmtime](https://wasmtime.dev/);
  LLM inference engines talk to aicirt via shared memory and semaphores
- [rLLM](rllm) - a reference implementation of an LLM inference engine
- [pyaici](pyaici) - a Python package for interacting with aicirt and running controllers
- [promptlib](promptlib) - a Python package that exposes API for easily creating and running DeclCtrl ASTs
  (will change to generate PyCtrl programs in the future)

And a number of sample/reference controllers:

- [yes/no](aici_abi/src/yesno.rs) and [uppercase](aici_abi/src/uppercase.rs) - small samples for aici_abi
- [PyCtrl](pyctrl) - an embedded Python 3 interpreter (using [RustPython](https://github.com/RustPython/RustPython)),
  which lets you write controllers in Python
- [JsCtrl](jsctrl) - an embedded JavaScript interpreter (using [QuickJS](https://bellard.org/quickjs/)),
  which lets you write controllers in JavaScript
- [DeclCtrl](declctrl) - a controller that interprets a simple JSON AST (Abstract Syntax Tree) to specify constraints

Everything above implemented in Rust, unless otherwise stated.

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
export AICI_API_BASE="https://aici.azurewebsites.net/v1/#key=wht_..."
```

Now, use query the model with or without AICI Controller:

```bash
./scripts/aici.sh infer --prompt "The answer to the ultimate question of life"
./scripts/aici.sh run pyctrl/samples/test.py
./scripts/aici.sh run declctrl/arg2.json
./scripts/aici.sh run --build aici_abi::uppercase
```

Run `./scripts/aici.sh -h` to see usage info.

If the server is running with Orca-2 13B model,
you can also run tests with `pytest` for the DeclCtrl, or with `./scripts/test-pyctrl.sh` for PyCtrl.

### Running local server

To run rLLM server, go to `rllm/` and run `./server.sh orca`.
This will run the inference server with Orca-2 13B model (which is expected by testcases).
You can also try other models, see [rllm/README.md](rllm/README.md) for details.

## Provided Controllers

We provide several controllers that you can use directly or as base for your own controller.

### Yes/no

The [yes/no controller](aici_abi/src/yesno.rs)
only allows the model to say "Yes" or "No" in answer to the question in the prompt.

```
$ ./scripts/sample-yesno.sh "Can orcas sing?"
will build yesno from /workspaces/aici/aici_abi/Cargo.toml
    Finished release [optimized + debuginfo] target(s) in 0.09s
built: /workspaces/aici/target/wasm32-wasi/release/yesno.wasm, 0.187 MiB
upload module... 191kB -> 668kB id:255ce305
[0]: tokenize: "Yes" -> [8241]
[0]: tokenize: "No" -> [3782]
[0]: tokenize: "\n" -> [13]
[DONE]

[Prompt] Can orcas sing?

[Response]
Yes
```

Note that the same effect can be achieved with PyCtrl and [10x less lines of code](pyctrl/samples/yesno.py).
This is just for demonstration purposes.

```
$ ./scripts/aici.sh run pyctrl/samples/yesno.py --prompt "Are dolphins fish?"
...
No
```

### Uppercase

The [uppercase controller](aici_abi/src/uppercase.rs) shows usage of the `FunctionalRecognizer` interface.
It forces every 4th letter of the model output to be uppercase.

```
$ ./scripts/sample-uppercase.sh
will build uppercase from /workspaces/aici/aici_abi/Cargo.toml
    Finished release [optimized + debuginfo] target(s) in 0.09s
built: /workspaces/aici/target/wasm32-wasi/release/uppercase.wasm, 0.193 MiB
upload module... 197kB -> 687kB id:4d3b70bf
[0]: user passed in 0 bytes
[0]: init_prompt: [1] ""
[0]: tokenize: "Here's a tweet:\n" -> [10605, 29915, 29879, 263, 7780, 300, 29901, 13]
[DONE]

[Prompt]

[Response] Here's a tweet:
I'm SO EXCITED! I'm GoinG toBe aMom!I'm GoinG toHaVeA BaBy!
```

Again, this could be done with PyCtrl and a simple regex.

### PyCtrl

The [PyCtrl](pyctrl) embeds [RustPython](https://github.com/RustPython/RustPython)
(a Python 3 language implementation) in the Wasm module together with native
primitives for specific kinds of output constraints:
fixed token output, regexps, LR(1) grammars, substring constrains etc.
Python code is typically only used lightly, for gluing the primitives together,
and thus is not performance critical.

There are [several samples](pyctrl/samples/) available.
The scripts use the [pyaici.server module](pyaici/server.py) to communicate with the AICI runtime
and use the native constraints.

To run a PyCtrl sample (using controller tagged with `pyctrl-latest`) use:

```bash
./scripts/aici.sh run pyctrl/samples/test.py
```

If you want to build it yourself, use:

```bash
./scripts/aici.sh run --build pyctrl pyctrl/samples/test.py
```

You will see the console output of the program.

### DeclCtrl

The [DeclCtrl](declctrl/src/declctrl.rs) exposes similar constraints
to PyCtrl, but the glueing is done via a JSON AST (Abstract Syntax Tree) and thus is
more restrictive.

There is no reason to use it as is, but it can be used as a base for other controller.

## Security

- `aicirt` runs in a separate process, and can run under a different user than the LLM engine
- Wasm modules are [sandboxed by Wasmtime](https://docs.wasmtime.dev/security.html)
- Wasm only have access to [`aici_host_*` functions](aici_abi/src/host.rs),
  implemented in [hostimpl.rs](aicirt/src/hostimpl.rs)
- `aicirt` also exposes a partial WASI interface; however almost all the functions are no-op, except
  for `fd_write` which shims file descriptors 1 and 2 (stdout and stderr) to print debug messages

In particular, Wasm modules cannot access the filesystem, network, or any other resources.
They also cannot spin threads or access any timers (this is relevant for Spectre/Meltdown attacks).

## Architecture

This AICI runtime is implemented in the [aicirt](aicirt) crate, while the binary AICI interface
is specified in the [aici_abi](aici_abi) crate.

The LLM engines are often implemented in Python, and thus the [pyaici](pyaici) Python packages provides
a class to spin up and communicate with `aicirt` process via POSIX shared memory and semaphores.
Using shared memory ensures there is very little work to be done on the Python side
(other than wrapping that memory as a tensor).

The [rllm](rllm) crate implements a reference LLM engine, which can be used for testing.
**The text below is outdated.**

The (harness)[harness] folder contains samples for using aicirt with different LLM engines:

- [HuggingFace Transformers](harness/run_hf.py), run with `./scripts/hf.sh`
- [vLLM script](harness/run_vllm.py), run with `./scripts/vllm.sh`
- [vLLM REST server](harness/vllm_server.py), run with `./scripts/server.sh`;
  the REST server is compatible with OpenAI and adds an endpoint for uploading Wasm modules;
  see [pyaici.rest](pyaici/rest.py) for an example on how it can be used

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
