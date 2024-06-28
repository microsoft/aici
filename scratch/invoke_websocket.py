# Connect to 
# localhost:4242/v1/sessions
# as a websocket client

import asyncio
import json
import websockets

async def invoke_websocket():
    uri = "ws://localhost:4242/v1/session"
    prefix = ' how are you my friend ? ' * 1000
    async with websockets.connect(uri) as websocket:
        data = {'type': 'create_prefix', 'name': 's', 'prefix': prefix, 'following': '',
                'sampling_params': {}},
        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        print(response)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(invoke_websocket())