#!/bin/sh

if [ "X$AICI_API_BASE" = "X" ] ; then
  export AICI_API_BASE="http://127.0.0.1:4242/v1/"
fi

curl -X POST "${AICI_API_BASE}run" \
-H "Content-Type: application/json" \
-d '{"controller": "llguidance", "controller_arg": {"grammar": {"grammars": [{"nodes": [{"Join": {"sequence": [1, 2]}}, {"String": {"literal": "2 + 2 = "}}, {"Join": {"sequence": [3]}}, {"Gen": {"body_rx": "[0-9]+", "stop_rx": " ", "lazy": true, "stop_capture_name": null, "temperature": 0.0}}], "rx_nodes": []}]}}, "prompt": "", "max_tokens": 3, "temperature": 0.0}'
