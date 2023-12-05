# AiciPython

This crate implements AI Controller Interface via an embedded Python interpreter
([RustPython](https://github.com/RustPython/RustPython)).

The Python interpreter does not include the full Python standard library, however
[parts are bundled](./Lib).
In addition to regular Python modules, the `aici` module is also [bundled](./aici-pylib).
It defines `AiciCallbacks` interface, which closely reflects the structure of native AICI interface.
`AiciAsync`, takes a async method and turns it into `AiciCallbacks` implementation;
this is typically invoked via `aici.start()`.

```python
import aici

async def sample():
    # initialization code
    print("I'm going in the logs!")
    # ... more initialization code, it has long time limit
    prompt = await aici.GetPrompt()
    # here we're out of initialization code - the time limits are tight

    # This appends the exact string to the output; similar to adding it to prompt
    await aici.FixedTokens("The word 'hello' in French is")

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
import aici

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
import aici

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
* tokenizer/detokenizer

You should limit the amount of Python code you run after generating tokens to a few lines.