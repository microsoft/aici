#!/bin/sh

for f in `find -name \*.py | xargs cat | grep -E '^(import |from \S+ import)' | awk '{print $2}' | sort | uniq` ; do
test -f $f.py && continue
test -f $f/__init__.py && continue
if test -f ../../RustPython/pylib/Lib/$f.py ; then
  echo cp ../../RustPython/pylib/Lib/$f.py $f.py
  continue
fi
echo "? $f"
done
