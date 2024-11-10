# AICI Uppercase

This folder provides a simple, hello-world-like AICI Controller
that you can clone to build your own controller.

When you clone it, make sure to keep [.cargo/config.toml](.cargo/config.toml)
as it sets up the linker flags for the Wasm target.

The [main.rs](src/main.rs) shows usage of the `FunctionalRecognizer` interface.
It forces every 4th letter of the model output to be uppercase.

```
$ ../../aici.sh run --build .
will build aici_uppercase from /workspaces/aici/uppercase/Cargo.toml
   Compiling aici_abi v0.1.0 (/workspaces/aici/aici_abi)
   Compiling aici_uppercase v0.1.0 (/workspaces/aici/uppercase)
    Finished release [optimized + debuginfo] target(s) in 1.81s
built: /workspaces/aici/target/wasm32-wasip1/release/aici_uppercase.wasm, 0.189 MiB
upload module... 193kB -> 675kB id:a4000d9b
[DONE]
[Response] Here's a tweet:
I'm SO EXCITED! I'm GoinG toBe aMom!I'm GoinG toHaVeA BaBy!
```

This is only meant as a sample - it could be done with [PyCtrl](../pyctrl) or
[JsCtrl](../jsctrl) and a simple regex.

## Yes/no controller

You can also take a look at the [yes/no controller](../aici_abi/src/yesno.rs), which
only allows the model to say "Yes" or "No" in answer to the question in the prompt.

```
$ ./scripts/sample-yesno.sh "Can orcas sing?"
+ echo 'Can orcas sing?'
+ ./aici.sh run --build aici_abi::yesno -
will build yesno from /workspaces/aici/aici_abi/Cargo.toml
   Compiling aici_abi v0.1.0 (/workspaces/aici/aici_abi)
    Finished release [optimized + debuginfo] target(s) in 0.58s
upload module... 192kB -> 671kB id:c65e78e9
[DONE]
[Response] Can orcas sing?

Yes
```

A similar effect can be achieved with PyCtrl and [10x less lines of code](../pyctrl/samples/yesno.py),
but it illustrates the raw token APIs.


```
$ ./aici.sh run pyctrl/samples/yesno.py
Running with tagged AICI Controller: pyctrl-latest
[0]: FIXED 'Are dolphins fish?\n'
[0]: GEN 'No'
[DONE]
[Response] Are dolphins fish?
No
```

## DeclCtrl

For a more full-fledged example, take a look at the [DeclCtrl](../declctrl/src/declctrl.rs).
