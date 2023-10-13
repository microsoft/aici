#!/bin/sh
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "aici_module": "8c7ee4c974f85548b16d997da3e87cb6034c7b29606ea1dcc889435b0370576e",
    "prompt": "42\n",
    "max_tokens": 70,
    "temperature": 0,
    "stream": true
  }'