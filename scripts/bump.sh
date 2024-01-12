#!/bin/sh

if [ "X`git status --porcelain --untracked-files=no`" != X ] ; then
  git status
  echo
  echo "*** You have local changes; cannot bump." 
  exit 1
fi

git pull || exit 1

eval `git describe --dirty --tags --match 'v[0-9]*' --always | sed -e 's/-.*//; s/v/v0=/; s/\./ v1=/; s/\./ v2=/'`
defl=0.0.0
if [ "X$v0" != X ] ; then
  defl=$v0.$v1.$(($v2 + 1))
fi
set -e
echo "Enter version [Enter = $defl; Ctrl-C to cancel]:"
read ver
if [ "X$ver" = "X" ] ; then
  ver="$defl"
else
  ver=$(echo "$ver" | sed -e 's/v//i')
fi
if echo "$ver" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' ; then
  :
else
  echo "Invalid version: $ver"
  exit 1
fi

set -x
git tag "v$ver"
git push --tags
git push
