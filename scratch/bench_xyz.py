"""
Benchmark on vLLM on a "x-y-z" style cache eviction request.
"""
import argparse
import asyncio
import json
from typing import List
import httpx

class App:
    def __init__(self, app_id: int, api_url: str, x: str, y: str, z: str):
        self.app_id = app_id
        self.api_url = api_url
        self.x = x
        self.y = y
        self.z = z

    async def post_http_request(self, prompt: str, max_tokens: int = 512, n: int = 1):
        headers = {
            "Content-Type": "application/json",
        }
        pload = {
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "model": "microsoft/Orca-2-13b",
        }

        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(self.api_url, headers=headers, json=pload)
            return response

    def get_responses(self, response: httpx.Response) -> List[str]:
        data = response.json()
        output = data["choices"]
        output = [d['text'] for d in output]
        return output

    def get_response(self, response: httpx.Response) -> str:
        outputs = self.get_responses(response)
        return outputs[0]

    async def run(self):
        async def _send(prompt, step, **kwargs):
            response = await self.post_http_request(prompt, **kwargs)
            output = self.get_response(response)
            print(f"App ID {self.app_id}, Step {step}: {len(output)=}")
            return output

        # Send a request with x + y + z -> t (output text store in var t)
        t = await _send(self.x + self.y + self.z, "t")
        # Send a request with x + z -> w (output text store in var w)
        w = await _send(self.x + self.z, "w")
        # Send a request with y + z -> u (output text store in var u)
        u = await _send(self.y + self.z, "u")
        # Send a request with x + u + z + y -> v (output text store in var v)
        v = await _send(self.x + self.y + self.z + self.y, "v")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--n", type=int, default=4)
    return parser.parse_args()


async def main():
    args = parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n

    print(f"Running {n} applications in parallel.")
    print(f"Sending requests to {api_url}.")

    apps = []
    for i in range(n):
        app = App(app_id=i + 1, api_url=api_url, x="2 + 2 = ", y="2 + 2 = ", z="2 + 2 = ")
        apps.append(asyncio.create_task(app.run()))

    await asyncio.gather(*apps)


if __name__ == "__main__":
    asyncio.run(main())
