# Frequently Asked Questions

## How does system prompt or chat mode work with AICI?

AICI interacts with models at the level of sequences of tokens.
Models themselves do not have a distinct input for "system prompt" or "chat message",
instead they are wrapped in model-specific tokens.
You need to find the model's "Instruction format", typically on model's page on HuggingFace.

For example, the [Orca-2-13b model](https://huggingface.co/microsoft/Orca-2-13b) has the following instruction format:
```
<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant
```

The [Mistral-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and [Mixtral-Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1), as well as [CodeLlama-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) models use:
```
[INST]{instruction}[/INST]
```

Intrestingly, `<|im_start|>` and `<|im_end|>` are special tokens, while `[INST]` and `[/INST]` are regular strings.

The start token (typically denoted `<s>`) is always implicit in AICI!

For example, for Orca model you can use the following:

```python
import pyaici.server as aici

system_message = "You are a helpful assistant."

async def ask(user_message: str):
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    await aici.FixedTokens(prompt)

async def main():
    await ask("What is the capital of Poland?")
    await aici.gen_tokens(max_tokens=10, store_var="capital")

aici.start(main())
```