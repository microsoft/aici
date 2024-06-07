#!/bin/sh

if [ "X$AZURE_GUIDANCE_URL" = "X" ] ; then
    if [ "X$AICI_API_BASE" = "X" ] ; then
        AICI_API_BASE="http://127.0.0.1:4242/v1/"
    fi
    AZURE_GUIDANCE_URL="$AICI_API_BASE"
fi
export AZURE_GUIDANCE_URL

TEST_SUFF=
# if $1 starts with :: then set it as TEST_SUFF
if [ "X$1" != "X" ] ; then
    if [ "X${1:0:2}" = "X::" ] ; then
        TEST_SUFF="$1"
        shift
    fi
fi

cd $(dirname $0)/../py/guidance
pytest --selected_model azure_guidance --durations=10 tests/models/test_azure_guidance.py$TEST_SUFF "$@"

