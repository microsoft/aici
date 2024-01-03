import subprocess
import requests
import ujson
import sys
import os
import re
from typing import Optional
from .endpoint import Endpoint

class AICI(Endpoint):
    # TODO remove this default base_url once we deploy a semi-permanent server
    def __init__(self, base_url="http://127.0.0.1:8080/v1/", wasm_runner_id=None, wasm_runner_path=None, wasm_runner_buildsh=None ):
        self.base_url = base_url

        if wasm_runner_id is None:
            if wasm_runner_path is None:
                if wasm_runner_buildsh is None:
                    raise RuntimeError("Must specify wasm_runner_id or wasm_runner_path or wasm_runner_buildsh")                
                wasm_runner_path = _compile_wasm(wasm_runner_buildsh)
            wasm_runner_id = _upload_wasm(self.base_url, wasm_runner_path)
        self.wasm_runner_id = wasm_runner_id

    def supportsAIVM(self):
        return True
    
    def run(self, prompt_plan):
        _submit_program(self.base_url, self.wasm_runner_id, prompt_plan, log=True)


def _compile_wasm(wasm_runner_buildsh, scriptargs=["build"]):
    # separate wasm_runner_buildsh into the script filename and the directory
    # containing the script
    script_dir = os.path.dirname(wasm_runner_buildsh)
    script_name = os.path.basename(wasm_runner_buildsh)
    
    r = subprocess.run(["sh", script_name].extend(scriptargs), cwd=script_dir)
    if r.returncode != 0:
        raise RuntimeError(f"error compiling aici promptlib module")
    
    file_path = script_dir + "/../target/strip.wasm"
    return file_path


def _upload_wasm(base_url, wasm_runner_path):    
    print("upload module... ", end="")
    with open(wasm_runner_path, "rb") as f:
        resp = requests.post(base_url + "aici_modules", data=f)
        if resp.status_code == 200:
            dd = resp.json()
            mod_id = dd["module_id"]
            print(
                f"{dd['wasm_size']//1024}kB -> {dd['compiled_size']//1024}kB id:{mod_id[0:8]}"
            )
            return mod_id
        else:
            raise RuntimeError(
                f"bad response to model upload: {resp.status_code} {resp.reason}: {resp.text}"
            )


def _submit_program(base_url, aici_module, aici_arg, temperature=0, max_tokens=200, n=1, log=False):
    json = {
        "model": "",
        "prompt": "",
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "stream": True,
        "aici_module": aici_module,
        "aici_arg": aici_arg,
    }
    resp = requests.post(base_url + "completions", json=json, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(
            f"bad response to completions: {resp.status_code} {resp.reason}: {resp.text}"
        )
    full_resp = []
    texts = [""] * n
    for line in resp.iter_lines():
        if line:
            decoded_line: str = line.decode("utf-8")
            if decoded_line.startswith("data: {"):
                d = ujson.decode(decoded_line[6:])
                full_resp.append(d)
                for ch in d["choices"]:
                    idx = ch["index"]
                    if idx == 0:
                        if log:
                            l = ch["logs"].rstrip("\n")
                            if "Previous WASM Error" in l:
                                raise "Bailing out due to WASM error: " + l
                        #else:
                        #    print(ch["text"], end="")
                    texts[idx] += ch["text"]
            #elif decoded_line == "data: [DONE]":
            #    print(" [DONE]")
            #else:
            #    print(decoded_line)

    if len(texts) == 1:
        print(texts[0])
    else:
        print(texts)
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/response.json"
    with open(path, "w") as f:
        ujson.dump(
            {"request": json, "texts": texts, "response": full_resp}, f, indent=1
        )
    #print(f"response saved to {path}")
