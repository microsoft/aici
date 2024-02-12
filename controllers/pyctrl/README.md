# PyCtrl

This crate implements AI Controller Interface by embedding [RustPython](https://github.com/RustPython/RustPython)
(a Python 3 language implementation) in the Wasm module together with native
primitives for specific kinds of output constraints:
fixed token output, regexps, LR(1) grammars, substring constrains etc.
Python code is typically only used lightly, for gluing the primitives together,
and thus is not performance critical.

There are [several sample scripts](samples/) available.
The scripts use the [pyaici.server module](../../py/pyaici/server.py) to communicate with the AICI runtime
and use the native constraints.

This is quite similar to [jsctrl](../jsctrl/README.md) but with Python instead of JavaScript.

## Usage

You can build, upload, and tag the PyCtrl Wasm module using the `aici.sh` script
(this assumes a [running server](../../README.md#build-and-start-rllm-server-and-aici-runtime)):

```bash
../aici.sh build . --tag pyctrl-latest
```

Then you can run a PyCtrl sample:

```bash
../aici.sh run --ctrl pyctrl-latest samples/test.py
```

You can also build and run in one step (without tagging):

```bash
../aici.sh run --build . samples/test.py
```

Either way, you will see the console output of the program.

By default, if you don't pass `--ctrl` to `aici.sh` but do pass a `.py` file,
it will download and use `gh:microsoft/aici/pyctrl`,
which is the [latest release](https://github.com/microsoft/aici/releases/latest) of PyCtrl.

The Python interpreter does not include the full Python standard library, however
[parts are bundled](./Lib).
In addition to regular Python modules, the `pyaici.server` module is also [bundled](../../py/pyaici/server.py).
It defines `AiciCallbacks` interface, which closely reflects the structure of native AICI interface.
`AiciAsync`, takes a async method and turns it into `AiciCallbacks` implementation;
this is typically invoked via `aici.start()`.

```python
import pyaici.server as aici

async def sample():
    # initialization code
    print("I'm going in the logs!")
    # ... more initialization code, it has long time limit

    # This appends the exact string to the output; similar to adding it to prompt
    await aici.FixedTokens("The word 'hello' in French is")

    # here we're out of initialization code - the time limits are tight

    # generate text (tokens) matching the regex
    french = await aici.gen_text(regex=r' "[^"]+"', max_tokens=5)
    # set a shared variable (they are returned as JSON and are useful with aici.fork())
    aici.set_var("french", french)

    await aici.FixedTokens(" and in German")
    # shorthand for the above
    await aici.gen_text(regex=r' "[^"]+"', store_var="german")

    await aici.FixedTokens("\nFive")
    # generates one of the strings
    # aici.gen_tokens() and gen_text() are the same, except for return type
    await aici.gen_tokens(options=[" pounds", " euros", " dollars"])

aici.start(sample())
```

## Backtracking

In LLMs tokens are generated one by one, and it's possible to cheaply remove a bunch
of recently generated tokens.
It's also possible to append more than one token at a time (see `aici.FixedTokens()` above).
The `aici.Label()` is used to mark a point in the generated sequence,
and the `following=` argument to `FixedTokens()` is used to backtrack.
For example:

```python
import pyaici.server as aici

async def backtracking():
    await aici.FixedTokens("The word 'hello' in")
    # mark the current position
    l = aici.Label()
    # append text at label (here the following= is superfluous)
    await aici.FixedTokens(" French is", following=l)
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="french", max_tokens=5)
    # now again append text at label - here following= is required
    await aici.FixedTokens(" German is", following=l)
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="german", max_tokens=5)

aici.start(backtracking())
```

This will generate the French word, store it in variable,
have the LLM forget about it, and generate the German word (plus store it).

Note that this happens sequentially, we can use `aici.fork()` to generate both words in parallel.

## Forking

The generation process can be forked into multiple branches (possibly more than two).
The branches can communicate through shared variables (`aici.set_var()` and `aici.get_var()`).

```python
import pyaici.server as aici

async def forking():
    await aici.FixedTokens("The word 'hello' in")
    # fork into three branches
    id = await aici.fork(3)
    # see which branch we're in
    if id == 0:
        # in first branch, we wait for the other two branches to finish
        french, german = await aici.wait_vars("french", "german")
        # append some text, based on what the other branches did
        await aici.FixedTokens(f"{french} is the same as {german}.")
        # and then generate some tokens
        await aici.gen_tokens(max_tokens=5)
    # the other two branches are similar to previous examples
    elif id == 1:
        await aici.FixedTokens(" German is")
        await aici.gen_tokens(regex=r' "[^"]+"', store_var="german", max_tokens=5)
    elif id == 2:
        await aici.FixedTokens(" French is")
        await aici.gen_tokens(regex=r' "[^"]+"', store_var="french", max_tokens=5)

aici.start(forking)
```

## Tokens, bytes, and strings

LLMs generate tokens. Each token is identified by a unique integer
(we generally do not use the string names for tokens).
Different models have different token sets (vocabularies), eg. 32000 tokens for Llama.
Each token corresponds to a sequence of bytes; these often are valid UTF-8 strings,
but not always (eg., some emojis or rare Unicode characters will be split across multiple tokens).
Because of this:
* the `aici.gen_tokens()` returns `list[int]`
* `aici.gen_text()` returns `str`, possibly with Unicode replacement characters (`�`);
  this may not do the right thing in presence of non-UTF-8 bytes
* shared variables are stored and returned as `bytes` (though when writing them, you can use `str`)

We may need to extend `re` with support for matching `bytes` not only `str` in future.


## Restrictions and compatibility

* you can't access files or network
* only parts of the standard library are included (though more modules are easily added)
* `re` module is available; all `str` methods are also available
* you can't `pip install`
* there is no multi-threading (but see `aici.fork()`)

RustPython is generally compatible with Python 3.

## Performance

Performance-critical code is implemented natively. This includes:

* `TokenSet` class
* `RegexConstraint` class
* `SubstrConstraint` class
* tokenizer/detokenizer

You should limit the amount of Python code you run after generating tokens to a few lines.