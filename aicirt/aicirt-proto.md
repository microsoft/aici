# Protocol between LLM inference engine and AICI-runtime

The LLM inference engine (LLM) and AICI-runtime (AICIrt) communicate via a
JSON messages sent over POSIX shared memory (SHM).
There are two alternative synchronization mechanisms for the SHM region:
* POSIX named semaphores
* `futex` on Linux/`__ulock` on macOS/`WaitOnAddress` on Windows ([issue](https://github.com/microsoft/aici/issues/42));
  this requires `--futex` flag to be passed to AICIrt
Regardless of the chosen synchronization mechanism, the message format is the same.

The LLM side of the interface is implemented in [comms.py](../pyaici/comms.py)
and in [iface.rs](../rllm/src/iface.rs).
The Python interface is outdated: [tracking issue](https://github.com/microsoft/aici/issues/43).

Two bi-direction message channels are used:
- the main channel - synchronous
- the side channel - asynchronous

