import subprocess
import ujson
import sys
import os

import pyaici.ast as ast
import pyaici.rest
import pyaici.util


def upload_wasm(prog="aici_ast_runner"):
    r = subprocess.run(["sh", "wasm.sh", "build"], cwd=prog)
    if r.returncode != 0:
        sys.exit(1)
    file_path = "target/opt.wasm"
    return pyaici.rest.upload_module(file_path)


def ask_completion(*args, **kwargs):
    res = pyaici.rest.completion(*args, **kwargs)
    print("\n[Prompt] " + res["request"]["prompt"] + "\n")
    for text in res["text"]:
        print("[Response] " + text + "\n")
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/response.json"
    with open(path, "w") as f:
        ujson.dump(res, f, indent=1)
    print(f"response saved to {path}")
    print("Usage:", res["usage"])
    print("Storage:", res["storage"])


def usage():
    print('Usage: ./scripts/upload.sh ["prompt text"|file.{txt,py,wasm,json}]')
    sys.exit(1)


def main():
    if len(sys.argv) <= 1:
        usage()

    fn = sys.argv[1]
    if fn.endswith(".txt"):
        pyaici.rest.log_level = 1
        prompt = open(fn).read()
        ask_completion(
            prompt=prompt,
            aici_module=None,
            aici_arg=None,
            ignore_eos=True,
            max_tokens=100,
        )
    elif fn.endswith(".py"):
        mod = upload_wasm("pyvm")
        pyaici.rest.log_level = 3
        arg = open(fn).read()
        if len(sys.argv) > 2:
            prompt = sys.argv[2]
        else:
            prompt = ""
        ask_completion(
            prompt=prompt,
            aici_module=mod,
            aici_arg=arg,
            ignore_eos=True,
            max_tokens=2000,
        )
    elif fn.endswith(".wasm"):
        mod = pyaici.rest.upload_module(fn)
        pyaici.rest.log_level = 3
        if len(sys.argv) > 2 and sys.argv[2]:
            arg = open(sys.argv[2]).read()
        else:
            arg = ""
        if len(sys.argv) > 3:
            prompt = sys.argv[3]
        else:
            prompt = ""
        ask_completion(
            prompt=prompt,
            aici_module=mod,
            aici_arg=arg,
            ignore_eos=True,
            max_tokens=2000,
        )
    elif fn.endswith(".json"):
        with open(fn) as f:
            arg = ujson.load(f)
        mod = upload_wasm()
        pyaici.rest.log_level = 1
        ask_completion(
            prompt=pyaici.util.orca_prompt(arg["prompt"]),
            aici_module=mod,
            aici_arg=arg,
            **arg["sampling_params"],
        )
    elif " " in fn:
        pyaici.rest.log_level = 1
        ask_completion(
            prompt=fn,
            aici_module=None,
            aici_arg=None,
            ignore_eos=True,
            max_tokens=100,
        )
    else:
        usage()


main()
