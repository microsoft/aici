# JsCtrl

This crate implements AI Controller Interface by embedding 
[QuickJS](https://bellard.org/quickjs/) (JavaScript (ES2023) interpreter)
via [rquickjs](https://github.com/DelSkayn/rquickjs)
in a Wasm module together with native
primitives for specific kinds of output constraints:
fixed token output, regexps, LR(1) grammars, substring constrains etc.
JavaScript code is typically only used lightly, for gluing the primitives together,
and thus is not performance critical.

There are [some sample scripts](./samples/) available.
The scripts use the [aici module](./samples/aici-types.d.ts) to communicate with the AICI runtime
and use the native constraints.

This is quite similar to [PyCtrl](../pyctrl/README.md) but with JavaScript instead of Python.
It is also smaller, at 1.3MiB without regex and CFG, 1.8MiB with regex, and 3.3MiB with regex and CFG.
For comparison, pyctrl is 14MiB.
Also, the [PyCtrl samples](../pyctrl/samples/) translate 1:1 to JsCtrl.

## Usage

To run a JsCtrl sample use:

```bash
../aici.sh run samples/hello.js
```

If you write your sample in TypeScript, compile it first with `tsc -p samples`.

If you want to build the interpreter yourself, use:

```bash
../aici.sh run --build . samples/hello.js
```

You will see the console output of the program.

