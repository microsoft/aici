# JsCtrl

This crate implements AI Controller Interface by embedding 
[QuickJS](https://bellard.org/quickjs/) (JavaScript (ES2023) interpreter)
via [rquickjs](https://github.com/DelSkayn/rquickjs)
in a Wasm module together with native
primitives for specific kinds of output constraints:
fixed token output, regexps, LR(1) grammars, substring constrains etc.
JavaScript code is typically only used lightly, for gluing the primitives together,
and thus is not performance critical.

There are [some sample scripts](ts/) available.
The scripts use the [aici module](ts/aici.ts) to communicate with the AICI runtime
and use the native constraints.

This is quite similar to [pyctrl](../pyctrl/README.md) but with JavaScript instead of Python.
It is also smaller, at 1.3MiB without regex and CFG, 1.8MiB with regex, and 3.3MiB with regex and CFG.
For comparision, pyctrl is 14MiB.

## Usage

To run a JsCtrl sample (using controller tagged with `jsctrl-latest`) use:

```bash
../aici.sh run ts/sample.js
```

If you write your sample in TypeScript, compile it first with `tsc -p ts`.

If you want to build the interpreter yourself, use:

```bash
../aici.sh run --build . ts/sample.js
```

You will see the console output of the program.

