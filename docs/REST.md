# REST APIs for AICI

AICI server exposes REST APIs for uploading and tagging Controllers (.wasm files),
and extends the "completion" REST APIs to allow for running the controllers.

## Uploading a Controller

To upload a controller, POST it to `/v1/controllers`.
Note that the body is the raw binary `.wasm` file, not JSON-encoded.
The `module_id` is just the SHA256 hash of the `.wasm` file.
The other fields in the response may or may not be returned:
the `wasm_size` is the input size in bytes, and `compiled_size` is the size of the compiled
Wasm file, `time` is the time it took to compile the Wasm file in milliseconds.

```json
// POST /v1/controllers
// ... binary of Wasm file ...
// 200 OK
{
  "module_id": "44f595216d8410335a4beb1cc530321beabe050817b41bf24855c4072c2dde2d",
  "wasm_size": 3324775,
  "compiled_size": 11310512,
  "time": 393
}
```

## Running a Controller

To run a controller, POST to `/v1/run`.
The `controller` parameter specifies the module to run, either the HEX `module_id`
or a tag name (see below).
The `controller_arg` is the argument to pass to the module; it can be either JSON object (it will be encoded as a string)
or a JSON string (which will be passed as is).
The `jsctrl` expects an argument that is the string, which is the program to execute.

```json
// POST /v1/run
{
  "controller": "jsctrl-latest",
  "controller_arg": "async function main() {\n    await $`Ultimate answer is to the life, universe and everything is `\n    await gen({ regex: /\\d\\d/ })\n}\n\nstart(main)\n"
}
```

```
200 OK
data: {"id":"run-cfa3ed5b-7be1-4e57-a480-1873ad096817","object":"initial-run","cr...
data: {"object":"run","forks":[{"index":0,"text":"Ultimate answer is to the life,...
data: {"object":"run","forks":[{"index":0,"text":"2","error":"","logs":"GEN \"42....
data: {"object":"run","forks":[{"index":0,"finish_reason":"aici-stop","text":" ",...
data: {"object":"run","forks":[{"index":0,"finish_reason":"aici-stop","text":"","...
data: [DONE]
```

There is `initial-run` object first, followed by zero or more `run` objects.
The final entry is string `[DONE]`.

Each `run` entry contains:
- `forks` - list of forks (sequences within the request)
- `usage` - information about the number of tokens processed and generated

Each fork contains:
- `text` - the result of the LLM; note that it will get confusing if you use backtracking 
  (AICI inserts additional `↩` characters to indicate backtracking)
- `logs` - console output of the controller
- `storage` - list of storage operations (that's one way of extracting the result of the controller);
  the `value` in `WriteVar` is hex-encoded byte string
- `error` - set when there is an error

The `usage` object contains:
- `sampled_tokens` - number of generated tokens
- `ff_tokens` - number of processed tokens (prompt, fast-forward, and generated tokens)
- `cost` - cost of the run (formula: `2*sampled_tokens + ff_tokens`; to be refined!)


```json
{
  "id": "run-cfa3ed5b-7be1-4e57-a480-1873ad096817",
  "object": "initial-run",
  "created": 1706571547,
  "model": "microsoft/Orca-2-13b"
}
```

```json
{
  "object": "run",
  "forks": [
    {
      "index": 0,
      "text": "Ultimate answer is to the life, universe and everything is 4",
      "error": "",
      "logs": "FIXED \"Ultimate answer is to the life, universe and everything is \"\nGEN-OPT {regex: /\\d\\d/}\nregex constraint: \"\\\\d\\\\d\"\ndfa: 160 bytes\n",
      "storage": []
    }
  ],
  "usage": {
    "sampled_tokens": 1,
    "ff_tokens": 15,
    "cost": 17
  }
}
```

```json
{
  "object": "run",
  "forks": [
    {
      "index": 0,
      "text": "2",
      "error": "",
      "logs": "GEN \"42\"\nJsCtrl: done\n",
      "storage": []
    }
  ],
  "usage": {
    "sampled_tokens": 2,
    "ff_tokens": 16,
    "cost": 20
  }
}
```

```json
{
  "object": "run",
  "forks": [
    {
      "index": 0,
      "finish_reason": "aici-stop",
      "text": " ",
      "error": "",
      "logs": "",
      "storage": []
    }
  ],
  "usage": {
    "sampled_tokens": 3,
    "ff_tokens": 17,
    "cost": 23
  }
}
```

```json
{
  "object": "run",
  "forks": [
    {
      "index": 0,
      "finish_reason": "aici-stop",
      "text": "",
      "error": "",
      "logs": "",
      "storage": []
    }
  ],
  "usage": {
    "sampled_tokens": 3,
    "ff_tokens": 17,
    "cost": 23
  }
}
```

## Tags

You can tag a `module_id` with one or more tags:

```json
// POST /v1/controllers/tags
{
  "module_id": "44f595216d8410335a4beb1cc530321beabe050817b41bf24855c4072c2dde2d",
  "tags": ["jsctrl-test"]
}
// 200 OK
{
  "tags": [
    {
      "tag": "jsctrl-test",
      "module_id": "44f595216d8410335a4beb1cc530321beabe050817b41bf24855c4072c2dde2d",
      "updated_at": 1706140462,
      "updated_by": "mimoskal",
      "wasm_size": 3324775,
      "compiled_size": 11310512
    }
  ]
}
```

You can also list all existing tags:

```json
// GET /v1/controllers/tags
// 200 OK
{
  "tags": [
    {
      "tag": "pyctrl-v0.0.3",
      "module_id": "41bc81f0ce56f2add9c18e914e30919e6b608c1eaec593585bcebd61cc1ba744",
      "updated_at": 1705629923,
      "updated_by": "mimoskal",
      "wasm_size": 13981950,
      "compiled_size": 42199432
    },
    {
      "tag": "pyctrl-latest",
      "module_id": "41bc81f0ce56f2add9c18e914e30919e6b608c1eaec593585bcebd61cc1ba744",
      "updated_at": 1705629923,
      "updated_by": "mimoskal",
      "wasm_size": 13981950,
      "compiled_size": 42199432
    },
    ...
  ]
}
```
