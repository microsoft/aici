

# This is working good
# curl -H "Content-Type: application/json" -X POST http://localhost:4242/v1/completions -d '{"prompt": "2 + 2 = ", "model": "microsoft/Orca-2-13b"}'
# curl -H "Content-Type: application/json" -X POST http://localhost:4242/v1/completions -d '{"prompt": "2 + 2 = ", "model": "microsoft/Orca-2-13b", "n": 32, "max_tokens": 1024}'

# Grab the models
# curl -X GET http://localhost:4242/v1/models

# Call the api_client.py to send the requests
set -x
cd `dirname $0`
HERE=`pwd`
cd $HERE/../py/vllm/examples/
# python3 api_client.py "$@"
# python3 api_client.py --port 4242 --n 1 --max_tokens 1024 
# python3 -m pdb -c continue api_client.py --port 4242 --n 1 "$@"
python3 api_client.py --port 4242 --n 1 "$@"