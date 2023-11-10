import subprocess
import ujson
import sys
import os
import unittest

import pyaici.ast
import pyaici.rest
import pyaici.util

prog = "aici_ast_runner"


def upload_wasm():
    r = subprocess.run(["sh", "wasm.sh", "build"], cwd=prog)
    if r.returncode != 0:
        sys.exit(1)
    file_path = prog + "/target/opt.wasm"
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


def main():
    ast = {
        "steps": pyaici.ast.json_to_steps(
            {
                "name": "",
                "valid": True,
                "description": "",
                "type": "foo|bar|baz|something|else",
                "address": {"street": "", "city": "", "state": "[A-Z][A-Z]"},
                "age": 1,
                "fraction": 1.5,
            }
        )
    }

    mod = upload_wasm()
    pyaici.rest.log_level = 1
    # read file named on command line if provided
    wrap = pyaici.util.codellama_prompt
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "test":
            pyaici.rest.log_level = 0
            pyaici.test.ast_module = mod
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(pyaici.test)
            runner = unittest.TextTestRunner()
            runner.run(suite)
        else:
            with open(sys.argv[1]) as f:
                ast = ujson.load(f)
            ask_completion(
                prompt=wrap(ast["prompt"]),
                aici_module=mod,
                aici_arg=ast,
                **ast["sampling_params"],
            )
    else:
        ask_completion(
            prompt=wrap("Say something about J.R.R. Tolkien"),
            # prompt=wrap("Write fib function in C, respond in code only"),
            aici_module=mod,
            aici_arg=ast,
            n=2,
            temperature=0.5,
            max_tokens=1000,
        )


main()
