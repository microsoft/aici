#!/bin/sh
(cd aici_ast_runner && ./wasm.sh)
mod=`cat tmp/runlog.txt |grep '^[a-f0-9]\{64\}$'`
curl http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "aici_module": "'$mod'",
    "prompt": "42\n",
    "max_tokens": 70,
    "temperature": 0,
    "stream": false,
    "aici_arg": 
    
{
  "steps": [
    {
      "Fixed": {
        "text": "I WAS about "
      }
    },
    {
      "Gen": {
        "max_tokens": 5,
        "rx": "\\d\\d"
      }
    },
    {
      "Fixed": {
        "text": " years and "
      }
    },
    {
      "Gen": {
        "max_tokens": 5,
        "rx": "\\d+"
      }
    }
  ]
}

  }'