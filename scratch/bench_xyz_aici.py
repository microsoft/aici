import argparse
import asyncio
import json
from typing import List
import httpx


class App:
    def __init__(self, app_id: int, api_url: str, program_path: str):
        self.app_id = app_id
        self.api_url = api_url
        # TODO: Possibly using a Jinja template or something to pass in the program.
        # For AICI, is there a way to pass in a program and call the function to run?
        self.program_path = program_path
        with open(self.program_path, "r") as f:
            self.program = f.read() 

    # Streaming response will get `data: {...}`
    
    async def post_http_request(self, controller_arg: str):
        headers = {
            # "Content-Type": "application/json",
        }
        pload = {
            "controller": "gh:microsoft/aici/pyctrl",
            "controller_arg": controller_arg,
            "prompt": "",
            "model": "microsoft/Orca-2-13b",
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=600) as client:
            # Send request
            print(f"Sent request: {self.app_id}")
            response = await client.post(self.api_url, headers=headers, json=pload)
            it = await response.aread()
            async for b in it:
                pass # ....
            return response

    
    async def run(self):
        async def _send(controller_arg, step, **kwargs):
            response = await self.post_http_request(controller_arg)
            output = response.text
            return output
    
        # Send a request to run the program
        output = await _send(self.program, "t")
        # print(f"App ID {self.app_id}, {output=}")
        print(f"Finished: app_id = {self.app_id}")
        pass
    
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--program_path", type=str, required=True)
    return parser.parse_args()

async def main():
    args = parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/run"
    n = args.n
    program_path = args.program_path

    print(f"Running {n} applications in parallel.")
    print(f"Sending requests to {api_url}.")

    apps = []
    for i in range(n):
        app = App(app_id=i + 1, api_url=api_url, program_path=program_path)
        # apps.append(asyncio.create_task(app.run()))
        apps.append(app.run())
    apps = [asyncio.create_task(app) for app in apps]

    # await asyncio.gather(*apps)
    # gather app, but for each that returns, print the output
    for app in asyncio.as_completed(apps):
        output = await app
        # print(output)


if __name__ == "__main__":
    asyncio.run(main())
