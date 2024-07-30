#!/bin/sh

if [ "X$AICI_API_BASE" = "X" ] ; then
  export AICI_API_BASE="http://127.0.0.1:4242/v1/"
fi

echo "using model name 'model'; can lead to 404"

curl -X POST "${AICI_API_BASE}chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "model",
  "messages": [{"role": "user", "content": "Hello, how are you?"}],
  "temperature": 0.7
}'
