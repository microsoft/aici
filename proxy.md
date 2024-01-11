# Client-side access to AICI

The [Artificial Intelligence Controller Interface (AICI)](https://github.com/microsoft/aici)
can be used to constrain output of an LLM in real time.
While the GPU is working on the next token of the output, the AICI runtime can use the CPU to
compute a user-provided constraint on the next token.
This adds minimal latency to the LLM generation.

## Setup

Install the `pyaici` package, export credentials, and see if the connection is working:

```bash
pip install git+https://github.com/microsoft/aici
export AICI_API_BASE="https://something.com/v1/#key=wht_..."
aici infer --max-tokens=10 --prompt="Answer to the Ultimate Question of Life, the Universe, and Everything is"
```

To test out the `pyctrl`, create `answer.py` file with:

```python
import pyaici.server as aici

async def main():
    await aici.FixedTokens("The ultimate answer to the universe is ")
    await aici.gen_text(regex=r'\d\d', max_tokens=2)

aici.start(main())
```

You can run it with `aici run answer.py`. Try `aici run --help` for available options.

You can use `aici --log-level=5 run answer.py` to see arguments to the REST requests,
if you want to do them yourself.
