# Client-side access to AICI

The [Artificial Intelligence Controller Interface (AICI)](https://github.com/microsoft/aici)
can be used to constrain output of an LLM in real time.
While the GPU is working on the next token of the output, the AICI VM can use the CPU to
compute a user-provided constraint on the next token.
This adds minimal latency to the LLM generation.

## Setup

Install the `pyaici` package, export credentials, and see if the connection is working:

```bash
pip install git+https://github.com/microsoft/aici
export AICI_API_BASE="https://something.com/v1/#key=wht_..."
pyaici infer --max-tokens=10 --prompt="Answer to the Ultimate Question of Life, the Universe, and Everything is"
```

## Testing

You can test the model by running the following command (where you replace `wht_...` with your API key):

```
curl -X POST https://aici.azurewebsites.net/v1/completions \
    -H 'content-type: application/json' \
    -H 'api-key: wht_...' \
    -d '{"model":"","prompt":"Hello world","max_tokens":5,
    "n":1,"temperature":0.0,"stream":true,"aici_module":null,
    "aici_arg":null,"ignore_eos":true}'
```

TODO: add more info
