

# This is working good
curl -H "Content-Type: application/json" -X POST http://localhost:4242/v1/completions -d '{"prompt": "2 + 2 = ", "model": "microsoft/Orca-2-13b"}'


# curl -X GET http://localhost:4242/v1/models