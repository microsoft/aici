llama_sys_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. You are concise.
"""


def llama_prompt(prompt: str) -> str:
    return f"[INST] <<SYS>>\n{llama_sys_prompt}\n<</SYS>>\n\n [/INST]</s>\n<s>[INST] {prompt} [/INST]\n"


def codellama_prompt(prompt: str) -> str:
    return f"[INST] {prompt} [/INST]\n"


system_message = "You are a helpful assistant."
orca_prefix = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n"
orca_suffix = "<|im_end|>\n<|im_start|>assistant\n"


def orca_prompt(prompt: str) -> str:
    return orca_prefix + prompt + orca_suffix
