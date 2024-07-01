# Connect to 
# localhost:4242/v1/sessions
# as a websocket client

import asyncio
import json
import websockets

async def invoke_websocket():
    uri = "ws://localhost:4242/v1/session"
    
    async with websockets.connect(uri, ping_timeout=600000, close_timeout=6000000) as websocket:
        prefix = ' how are you my friend ? '
        data = {'type': 'create_prefix', 'name': 's', 'prefix': prefix, 'following': '',
                'sampling_params': {}},
        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        print(response)


        prefix = ' I am good thank you. '
        data = {'type': 'create_prefix', 'name': 't', 'prefix': prefix, 'following': 's',
                'sampling_params': {}},
        await websocket.send(json.dumps(data))
        response = await websocket.recv()
        print(response)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(invoke_websocket())