import argparse
import asyncio
import json
import random
from typing import List
import httpx

class App:
    def __init__(self, app_id: int, api_url: str, n: int, prompt_len: int = 1024):
        self.app_id = app_id
        self.api_url = api_url
        self.n = n
        self.prompt_len = prompt_len

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
        async with httpx.AsyncClient(timeout=60000) as client:
            # Send request
            print(f"Sent request: {self.app_id}")
            response = await client.post(self.api_url, headers=headers, json=pload)
            # Check the status code
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return
            return response
    
    async def run(self):
        # Send n requests sequentially to the server
        for i in range(self.n):
            prompt = f" a" * self.prompt_len
            response = await self.post_http_request(prompt, max_tokens=1024)
            print(f"Received response: {self.app_id}")
        

# Run the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/v1/completions"
    app = App(app_id=0, api_url=api_url, n=args.n)
    asyncio.run(app.run())

