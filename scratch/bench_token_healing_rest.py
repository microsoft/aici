import argparse
import asyncio
import json
import random
from typing import List
import httpx

class App:
    def __init__(self, app_id: int, api_url: str, n: int):
        self.app_id = app_id
        self.api_url = api_url
        self.n = n

    # Streaming response will get `data: {...}`
    async def post_http_request(self, prompt: str, max_tokens: int) -> List[str]:
        headers = {
            # "Content-Type": "application/json",
        }
        pload = {
            "prompt": prompt,
            "model": "microsoft/Orca-2-13b",
            "stream": True,
            "max_tokens": 1024,
        }

        tokens = []
        async with httpx.AsyncClient(timeout=600) as client:
            # Send request
            print(f"Sent request: {self.app_id}")
            response = await client.post(self.api_url, headers=headers, json=pload)
            # TODO: This is a streaming request ....
            it = await response.aiter_lines()
            
            i = 0
            async for b in it:
                i += 1
                print(i, b)
                tokens.append(b)
                if i >= max_tokens:
                    break
        return tokens

    async def run(self):
        prompt = ""
        for i in range(self.n):
            # Extend the context to some amount of tokens
            ntok_to_gen = random.randint(2, 10)
            ntok_to_heal = random.randint(1, 3)
            toks0 = await self.post_http_request(prompt, ntok_to_gen + ntok_to_heal)
            
            # Backtrack some tokens in the prompt
            prompt += "".join(toks0[: -ntok_to_heal]) # TODO: Determine what does the token returns
            # May need to tokenize the input back to tokens?
            n_tok_healed = random.randint(1, 3)
            prompt += "".join([" a"] * n_tok_healed) # make the token healing happen
            pass
        return prompt
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--n", type=int, default=4)
    return parser.parse_args()

async def main():
    args = parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/run"
    n = args.n 

    app = App(app_id=1, api_url=api_url, n=n)   
    await app.run()
    pass

if __name__ == "__main__":
    asyncio.run(main())