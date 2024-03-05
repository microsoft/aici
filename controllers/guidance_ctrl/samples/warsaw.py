import pyaici.server as aici

system_message = "You are a helpful assistant."

async def fixed(user_message: str):
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    await aici.FixedTokens(prompt)

async def main():
    await fixed("What is the capital of Poland?")
    await aici.gen_tokens(max_tokens=10, store_var="capital")

aici.start(main())
