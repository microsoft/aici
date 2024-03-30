#!/bin/sh

export AICI="$(cd `dirname $0`; pwd)/aici.sh --all-prefixes"

case "$1" in
  *aici-controllers-wasm32-wasip2-*.tar.?z)
    mkdir -p tmp/aici-controllers
    tar --strip-components=1 -xf "$1" -C tmp/aici-controllers
    if test -f tmp/aici-controllers/tag.sh ; then
        cd tmp/aici-controllers
        ./tag.sh --latest
        rm -rf tmp/aici-controllers
    else
        echo "No tag.sh found in tmp/aici-controllers"
        exit 1
    fi
    ;;
  *)
    echo "Usage: $0 aici-controllers-wasm32-wasip2-....tar.[xz|gz]"
    exit 1
    ;;
esac
