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
    await aici.gen_text(options=[" pounds", " euros", " dollars"])

aici.start(sample())
```

There is also `aici.fork()` for generating multiple sequences in parallel,
`aici.Label()` and `FixedTokens(following=...)` for backtracking,
and `aici.wait_vars()` for waiting in fork branches until variables are set.

