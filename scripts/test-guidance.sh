#!/bin/sh

if [ "X$AZURE_GUIDANCE_URL" = "X" ] ; then
    if [ "X$AICI_API_BASE" = "X" ] ; then
        AICI_API_BASE="http://127.0.0.1:4242/v1/"
    fi
    AZURE_GUIDANCE_URL="$AICI_API_BASE"
fi
export AZURE_GUIDANCE_URL

cd $(dirname $0)/../py/guidance
pytest --selected_model azure_guidance tests/models/test_azure_guidance.py
