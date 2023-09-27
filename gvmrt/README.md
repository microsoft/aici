# GuidanceVM Runtime (gvmrt)

Multi-threaded wasmtime-based runner.

## Scenario

- vLLM starts GvmRt 
- user sends request to vLLM: store_wasm(wasm_binary, metadata) -> wasm_id
- vLLM handles this request by asking GvmRt
- user sends request to vLLM: complete(prompt, wasm_input, wasm_id)
- vLLM asks MT to start instance of wasm_id, with given wasm_input, prompt -> instance_id
- 

## Comms channels

- `cmd`, JSON, size = 8M, vLLM -> MT
- `resp`, JSON, size = 8M, MT -> vLLM
- `binresp`, Binary, size = 16M, MT -> vLLM

