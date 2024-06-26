"""
Benchmark on vLLM on a "x-y-z" style cache eviction request.
"""
import argparse
import asyncio
import json
from typing import List
import requests
import httpx


# Write a simple query to open ai chat
# curl -H "Content-Type: application/json" -X POST http://localhost:4242/v1/completions -d '{"prompt": "2 + 2 = ", "model": "microsoft/Orca-2-13b"}'


async def post_http_request(prompt: str, api_url: str, max_tokens: int = 512, n:int=1):
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
        response = await client.post(api_url, headers=headers, json=pload)
        return response

def get_responses(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["choices"]
    output = [d['text'] for d in output]
    return output


def get_response(response: requests.Response) -> str:
    outputs = get_responses(response)
    # TODO: Get the time / usage of time to come back
    return outputs[0]


async def app(api_url, x: str, y: str, z: str):

    async def _send(prompt, **kwargs):
        response = await post_http_request(prompt, api_url, **kwargs)
        return get_response(response)

    # Send a request with x + y + z -> t (output text store in var t)
    t = await _send(x + y + z) 
    print(t)
    # Send a request with x + z -> w (output text store in var w)
    w = await _send(x + z)
    print(w)
    # Send a request with y + z -> u (output text store in var u)
    u = await _send(y + z)
    print(u)
    # Send a request with x + u + z + y -> v (output text store in var v)
    v = await _send(x + u + z + y)
    print(v)
    
    return

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

    # Run `n` of these application in parallel
    x = "2 + 2 = "
    y = "2 + 2 = "
    z = "2 + 2 = "
    
    apps = []
    for i in range(n):
        apps.append(asyncio.create_task(app(api_url, x, y, z)))
    
    # Wait for all the applications to finish   
    await asyncio.gather(*apps)


if __name__ == "__main__":
    asyncio.run(main())