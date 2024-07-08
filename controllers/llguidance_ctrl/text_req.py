import requests

url = 'http://localhost:4242/v1/completions'

headers = {
    'Content-Type': 'application/json',
}

data = {
    'model': 'model',
    'prompt': 'Once upon a time,',
    'max_tokens': 50,
    'temperature': 0,
    'stream': True  
}

# read tmp/prompt.txt
with open('tmp/prompt.txt', 'r') as file:
    data['prompt'] = file.read()

response = requests.post(url, headers=headers, json=data, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
