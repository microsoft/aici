#!/bin/sh
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "prompt": "Say this is a test",
    "max_tokens": 7,
    "temperature": 0,
    "stream": true
  }'

