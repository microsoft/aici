# AICI Runtime (aicirt)

Multi-threaded wasmtime-based runner.

```mermaid
graph TD
    User1 <-- HTTP --> vLLM
    User2 <-- HTTP --> vLLM
    UserN <-- HTTP --> vLLM["vLLM Server<br>(batching)"]
    vLLM <-- CUDA/pytorch --> GPU
    vLLM <-- POSIX SHM --> aicirt[AICI-runtime]
    aicirt <-- Sockets+SHM --> Worker1[Worker1<br>Running Wasm]
    aicirt <-- Sockets+SHM --> Worker2[Worker2<br>Running Wasm]
    aicirt <-- Sockets+SHM --> WorkerM[WorkerM<br>Running Wasm]
```

```mermaid
sequenceDiagram
    actor User
    participant GPU
    participant vLLM
    participant aicirt as AICI-runtime
    vLLM -->> GPU: Model
    User -->> vLLM: Request (Prompt + Wasm)
    vLLM -->>+ aicirt: Prompt + Wasm
    aicirt -->>- vLLM: logit bias 1
    vLLM -->>+ GPU: Prompt
    vLLM -->> GPU: logit bias 1
    GPU -->> vLLM: token 1
    vLLM -->>+ aicirt: token 1
    vLLM -->> User: token 1
    aicirt -->>- vLLM: logit bias 2
    vLLM -->> GPU: logit bias 2
    GPU -->>- vLLM: token 2
    vLLM -->> User: token 2
```

Below is process structure.

- dotted arrow from A to B indicates that A sends requests to B (and gets responses)
- solid arrow from A to B indicates that A spawns (forks) B
- `spawner` is a special process, forked from `aicirt` at the beginning;
  for every user requests it spawns a process for top-level controller and a `common state` process 
  for handling shared state between
  all controller instances for that request (they can talk to the `common state` process)
- the top-level constraint can spawn more constraints, which can spawn yet more;
  `aicirt` has a direct connection to all these constraints though

```mermaid
graph TD
    vLLM ---> aicirt[AICI-runtime]
    vLLM -..-> aicirt
    aicirt -..-> spawner
    aicirt -..-> A0((A0))
    aicirt -..-> A1((A1))
    aicirt -..-> A2((A2))
    aicirt -..-> A3((A3))
    aicirt -..-> A4((A4))
    aicirt ---> spawner
    spawner --> A0
    spawner --> CommsA[Common state for A]
    subgraph User Request A
      A0 --> A1
      A0 --> A2
      A2 --> A3
      A2 --> A4
      A0 -..-> CommsA
      A1 -..-> CommsA
      A2 -..-> CommsA
      A3 -..-> CommsA
      A4 -..-> CommsA
    end
    aicirt -..-> B0((B0))
    aicirt -..-> B1((B1))
    spawner --> B0
    spawner --> CommsB[Common state for B]
    subgraph User Request B
      B0 -..-> CommsB
      B1 -..-> CommsB
      B0 --> B1
    end
```