import subprocess
import requests
import json
import sys
import os
import re
from typing import Optional

import pyaici.rest as aici_rest

class AICI:
    def __init__(self, base_url=None, wasm_runner_id=None, wasm_runner_path=None, wasm_runner_buildsh=None ):
        self.base_url = base_url

        if wasm_runner_id is None:
            if wasm_runner_path is None:
                if wasm_runner_buildsh is None:
                    raise RuntimeError("Must specify wasm_runner_id or wasm_runner_path or wasm_runner_buildsh")                
                wasm_runner_path = _compile_wasm(wasm_runner_buildsh)
            wasm_runner_id = _upload_wasm(self.base_url, wasm_runner_path)
        self.wasm_runner_id = wasm_runner_id
    
    def run(self, prompt_plan):
        return _submit_program(self.base_url, self.wasm_runner_id, prompt_plan, log=True)


def _compile_wasm(wasm_runner_buildsh, scriptargs=["build"]):
    # separate wasm_runner_buildsh into the script filename and the directory
    # containing the script
    script_dir = os.path.dirname(wasm_runner_buildsh)
    script_name = os.path.basename(wasm_runner_buildsh)
    
    r = subprocess.run(["sh", script_name].extend(scriptargs), cwd=script_dir)
    if r.returncode != 0:
        raise RuntimeError(f"error compiling aici promptlib module")
    
    file_path = script_dir + "/target/strip.wasm"
    return file_path


def _upload_wasm(base_url, wasm_runner_path):    
    print("upload module... ", end="")
    with open(wasm_runner_path, "rb") as f:
        resp = requests.post(base_url + "aici_modules", data=f)
        if resp.status_code == 200:
            d = resp.json()
            dd = d["data"]
            mod_id = dd["module_id"]
            print(
                f"{dd['wasm_size']//1024}kB -> {dd['compiled_size']//1024}kB id:{mod_id[0:8]}"
            )
            return mod_id
        else:
            raise RuntimeError(
                f"bad response to model upload: {resp.status_code} {resp.reason}: {resp.text}"
            )


def _submit_program(base_url, aici_module, aici_arg, temperature=0, max_tokens=None, log=False):
    return aici_rest.run_controller(controller=aici_module, controller_arg=aici_arg, temperature=temperature, max_tokens=max_tokens, base_url=base_url)
