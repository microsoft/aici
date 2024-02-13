# Protocol between LLM inference engine and AICI-runtime

The LLM inference engine (LLM) and AICI-runtime (AICIrt) communicate via a
JSON messages sent over POSIX shared memory (SHM).
The function of AICIrt is to spin processes for each sequence, start Wasm controllers inside them,
and collect the results for the LLM.

There are two alternative synchronization mechanisms for the SHM region:

- POSIX named semaphores
- `futex` on Linux/`__ulock` on macOS/`WaitOnAddress` on Windows ([issue](https://github.com/microsoft/aici/issues/42));
  this requires `--futex` flag to be passed to AICIrt

Regardless of the chosen synchronization mechanism, the message format is the same.

The LLM side of the interface is implemented in [comms.py](../py/pyaici/comms.py)
and in [iface.rs](../rllm/rllm-base/src/iface.rs).

Two bi-directional message channels are used:

- the main channel - synchronous
- the side channel - asynchronous

The generation of text in an LLM occurs in steps.
At each step, there is zero or more active sequences being processed.
The LLM computes logits (scores, later turned into probabilities) for every possible token for each sequence.
Then a single token is sampled for each of these sequences and appended.

The main channel is used synchronously with steps:

- the LLM asks AICIrt to execute `pre_process` callback of all current sequences
  and report which one can run and which need to be suspended for this step
  (they are waiting for something)
- the LLM schedules some of the non-supended sequences to compute logits for
- the LLM informs AICIrt about the scheduled sequences;
  AICIrt starts `mid_process` callback for all scheduled sequences
  that will return logit biases etc.
- the LLM starts computing logits
- AICIrt sends the logit biases to the LLM
- LLM adds computed logits and biases and samples tokens
- LLM asks AICIrt to execute `post_process` callback with the newly sampled tokens;
  the response may require the LLM to stop some sequences

For performance reasons, the `post_process` and `pre_process` of the next step
are merged into one call from LLM to AICIrt called `post_pre_process`.

In the main channel above the response to each AICIrt command comes before the next command.
There are also tight time limits on each request, as to avoid slowing down token generation.
In the side channel, a number of requests can be pending, and the time limits are much more relaxed.
The side channel is used for:

- uploading a new Wasm controller
- tagging an uploaded controller
- listing tags
- instantiating a controller for a given request - this happens before the request sees the GPU

## Main channel messages

The `ping` command is used to check if the AICIrt is alive.

```json
{"op":"ping"}
// response
{"type":"ok","data":{"pong":1}}
```

The requests always have an `op` field, and the responses always have a `type` field,
which is either `"ok"` or `"error"`, as well as a `data` field.

The `tokens` command gives the size of the vocabulary of the loaded tokenizer.

```json
{"op":"tokens"}
// response
{"type":"ok","data":{"vocab_size":32003}}
```

After the initial exchange, the LLM uses side channel to upload and instantiate Wasm controller
for the request (see below).
Once instantiated, the controller needs to be assigned to a sequence.
This is done with the `post_pre_process` command, which combines `pre_process` and `post_process`.
Here, the `post_ops` is empty, since this is the first call,
and `pre_ops` assigns controller `"run-062a793f-a83f-4198-a792-9dfc39f623a6"` to sequence `2`.
It also says that since last call to `post_pre_process` no sequences were disposed off (freed).

```json
{
  "op": "post_pre_process",
  "post_ops": [],
  "pre_ops": [
    { "id": 2, "req_id": "run-062a793f-a83f-4198-a792-9dfc39f623a6" }
  ],
  "freed": []
}
```

There is no response for `post` phase,
but for `pre` phase there is response for sequence number `2`.
The response indicates that there is no error, the sequence is not suspended,
it should not fork, and which tokens should be added.
The `logs` field contains the console output of the Wasm controller,
and the `micros` field contains the time it took to run the controller.
The `storage` field contains a list of executed storage commands.
This closely mirrors [REST API responses](REST.md).

```json
{
  "type": "ok",
  "data": {
    "post_seqs": {},
    "pre_seqs": {
      "2": {
        "error": "",
        "result": {
          "suspend": false,
          "num_forks": 1,
          "ff_tokens": [
            29965, 1896, 6490, 1234, 338, 304, 278, 2834, 29892, 19859, 322,
            4129, 338, 29871
          ]
        },
        "storage": [],
        "logs": "FIXED \"Ultimate answer is to the life, universe and everything is \"\nGEN-OPT {regex: /\\d\\d/}\nregex constraint: \"\\\\d\\\\d\"\ndfa: 160 bytes\n",
        "micros": 280
      }
    }
  }
}
```

Next, the LLM asks AICIrt to execute `mid_process` for sequence `2`;
it also says that sequence `2` is not a clone of an existing sequence
(the field can be also skipped altogether).
The order of sequences in `ops` is important: the logit bias will be returned
in shared memory in the same order as the sequences are passed in.

```json
{ "op": "mid_process", "ops": [{ "id": 2, "clone_id": null }] }
```

The response is similar to the one for `post_pre_process`, however while there is no specific `result`
in the JSON, there is logit bias in the shared memory region.

```json
{
  "type": "ok",
  "data": {
    "seqs": {
      "2": {
        "error": "",
        "result": null,
        "storage": [],
        "logs": "",
        "micros": 90
      }
    },
    "num_seqs": 1
  }
}
```

Next, `post_pre_process` is called again, this time with `post_ops` filled in:
it indicates that the sequence `2` has been advanced by 1 token,
and there was no backtracking.
The `pre_ops` is empty, meaning no new sequences are started.
However, the `pre_process` phase will be still called for all sequences that were
not freed.

```json
{
  "op": "post_pre_process",
  "post_ops": [{ "id": 2, "tokens": [29946], "backtrack": 0 }],
  "pre_ops": [],
  "freed": []
}
```

The response contains result of running the `post_process` and `pre_process` for sequence `2`.

```json
{
  "type": "ok",
  "data": {
    "post_seqs": {
      "2": {
        "error": "",
        "result": { "stop": false },
        "storage": [],
        "logs": "",
        "micros": 50
      }
    },
    "pre_seqs": {
      "2": {
        "error": "",
        "result": { "suspend": false, "num_forks": 1, "ff_tokens": [] },
        "storage": [],
        "logs": "",
        "micros": 10
      }
    }
  }
}
```

A similar exchange follows several more times.
Eventually, the `post_pre_process` will indicate that the sequence `2` should be stopped:

```json
{
  "type": "ok",
  "data": {
    "post_seqs": {
      "2": {
        "error": "",
        "result": { "stop": true },
        "storage": [],
        "logs": "",
        "micros": 30
      }
    },
    "pre_seqs": {
      "2": {
        "error": "",
        "result": { "suspend": false, "num_forks": 1, "ff_tokens": [] },
        "storage": [],
        "logs": "",
        "micros": 10
      }
    }
  }
}
```

In the next round, the LLM will tell AICIrt to dispose of the sequence `2`:

```json
{
  "op": "post_pre_process",
  "post_ops": [...],
  "pre_ops": [...],
  "freed": [2]
}
```

## Side channel messages

Here's a side request to instantiate a Wasm controller.
It has a randomly assigned `$rid` - this will be used in response.
The responses can come out of order, so it's important to keep this unique.
The request also includes information about the calling user (for logging etc.).
Just like main channel request, the `op` field indicates the kind of operation to run.

For the `instantiate` we pass the HTTP request ID (generated randomly by LLM),
the prompt (typically just the start symbol of the model if any),
the module ID, and the module argument.
The last two correspond to `controller` and `controller_arg` REST API fields.

```json
{
  "$rid": "0aae92c8-e415-4efd-947b-361a8573020c",
  "$auth": { "user": "localhost", "is_admin": true },
  "op": "instantiate",
  "req_id": "run-062a793f-a83f-4198-a792-9dfc39f623a6",
  "prompt": [1],
  "module_id": "jsctrl-latest",
  "module_arg": "async function main() {\n    await $`Ultimate answer is to the life, universe and everything is `\n    await gen({ regex: /\\d\\d/ })\n}\n\nstart(main)\n"
}
```

The response is pretty much empty, but note the matching `$rid`.

```json
{ "type": "ok", "data": {}, "$rid": "0aae92c8-e415-4efd-947b-361a8573020c" }
```

### Uploading and tagging controllers

The module to upload has to base64-encoded (unlike in the REST API where it's sent as binary).

```json
{
  "$rid": "61141b8b-9a85-4859-9c92-bf189920426f",
  "$auth": { "user": "localhost", "is_admin": true },
  "op": "mk_module",
  "binary": "AGFzbQEAAAABsAEXYAF/AGA.........W9yeSsPbXV0YWJsZS1nbG9iYWxzKwhzaWduLWV4dCsHc2ltZDEyOA=="
}
```

The returned `module_id` is sha256 of the Wasm of the module.
Compilation time is given in milliseconds (it might have used more than one core though).

```json
{
  "$rid": "61141b8b-9a85-4859-9c92-bf189920426f",
  "type": "ok",
  "data": {
    "module_id": "79c8dcb829ab3c0516524a0c2b37e5d8606b1986e39214da5d06820179465b2a",
    "wasm_size": 3301396,
    "compiled_size": 11258152,
    "time": 409
  }
}
```

The `set_tags` command is used to tag the uploaded module, with one or more tags.

```json
{
  "$rid": "e83d7f8a-8568-4f24-a82c-e2021b2c8cdd",
  "$auth": { "user": "localhost", "is_admin": true },
  "op": "set_tags",
  "module_id": "79c8dcb829ab3c0516524a0c2b37e5d8606b1986e39214da5d06820179465b2a",
  "tags": ["jsctrl-latest", "jsctrl-2024-01-30-2145"]
}
```

You will notice that the response contains `updated_by` fields, which are derived
from `$auth` field of the request.

```json
{
  "$rid": "e83d7f8a-8568-4f24-a82c-e2021b2c8cdd",
  "type": "ok",
  "data": {
    "tags": [
      {
        "tag": "jsctrl-latest",
        "module_id": "79c8dcb829ab3c0516524a0c2b37e5d8606b1986e39214da5d06820179465b2a",
        "updated_at": 1706651157,
        "updated_by": "localhost",
        "wasm_size": 3301396,
        "compiled_size": 11258152
      },
      {
        "tag": "jsctrl-2024-01-30-2145",
        "module_id": "79c8dcb829ab3c0516524a0c2b37e5d8606b1986e39214da5d06820179465b2a",
        "updated_at": 1706651157,
        "updated_by": "localhost",
        "wasm_size": 3301396,
        "compiled_size": 11258152
      }
    ]
  }
}
```

There is also a command to list (all) tags:

```json
{
  "$rid": "bb0db23d-42db-4467-9c64-c39e3a019662",
  "$auth": { "user": "localhost", "is_admin": true },
  "op": "get_tags"
}
```

With similar response, where the tags are ordered by `updated_at` field.

```json
{
  "$rid": "bb0db23d-42db-4467-9c64-c39e3a019662",
  "type": "ok",
  "data": {
    "tags": [
      {
        "tag": "jsctrl-2024-01-30-2152",
        "module_id": "553732d5c3e8cf7086b2a12054001eb7c1143616a3bf9118715dee60de31053c",
        "updated_at": 1706651547,
        "updated_by": "localhost",
        "wasm_size": 3301391,
        "compiled_size": 11258144
      },
      {
        "tag": "jsctrl-latest",
        "module_id": "553732d5c3e8cf7086b2a12054001eb7c1143616a3bf9118715dee60de31053c",
        "updated_at": 1706651547,
        "updated_by": "localhost",
        "wasm_size": 3301391,
        "compiled_size": 11258144
      },
      {
        "tag": "jsctrl-2024-01-30-2145",
        "module_id": "79c8dcb829ab3c0516524a0c2b37e5d8606b1986e39214da5d06820179465b2a",
        "updated_at": 1706651157,
        "updated_by": "localhost",
        "wasm_size": 3301396,
        "compiled_size": 11258152
      },
      ...
      {
        "tag": "declctrl-2024-01-11-2305",
        "module_id": "1abcea7075d4435966dd789c0e1a7c5c17da86161fd912b90e0b452a6c0cc6f1",
        "updated_at": 1705014317,
        "updated_by": "localhost",
        "wasm_size": 3726180,
        "compiled_size": 11412408
      }
    ]
  }
}
```
