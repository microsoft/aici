#!/bin/sh

if [ "X$AZURE_GUIDANCE_URL" = "X" ] ; then
    if [ "X$AICI_API_BASE" = "X" ] ; then
        AICI_API_BASE="http://127.0.0.1:4242/v1/"
    fi
    AZURE_GUIDANCE_URL="$AICI_API_BASE"
fi
export AZURE_GUIDANCE_URL

FILES="tests/need_credentials/test_azure_guidance.py tests/model_integration/test_greedy.py"

cd $(dirname $0)/../py/guidance

if [ "X$1" != "X" ] ; then
    if [ "X${1:0:2}" = "X::" ] ; then
        FILES="tests/need_credentials/test_azure_guidance.py$1"
        shift
        pytest --selected_model azure_guidance --durations=10 $FILES "$@"
        exit $?
    fi

    if [ "X$1" = "X--ll" ] ; then
        shift
        pytest tests/unit/test_ll.py "$@"
        exit $?
    fi

    if [ "X$1" = "X--server" ] ; then
        shift
        pytest --selected_model azure_guidance --durations=10 $FILES "$@"
        exit $?
    fi
fi

function runtest() {
    pytest "$@"
    if [ $? -ne 0 -a $? -ne 5 ] ; then
        :
        exit 1
    fi
}

# quick tests first
runtest tests/unit/test_ll.py "$@"
runtest tests/unit "$@"
runtest --selected_model azure_guidance --durations=10 $FILES "$@"
runtest tests/model_integration "$@"
