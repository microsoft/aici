#!/usr/bin/env python
import subprocess
import requests
import ujson
import sys
import os
import re
from typing import Optional

base_url = "http://127.0.0.1:8080/v1/"
prog = "aici_ast_runner"

ast = {
    "steps": [
        {"Fixed": {"text": "I am about "}},
        {"Gen": {"max_tokens": 10, "rx": r"\d+"}},
        {"Fixed": {"text": " years and "}},
        {"Gen": {"max_tokens": 10, "rx": r"\d+"}},
        {"Fixed": {"text": " months."}},
    ]
}


def gen(
    *,
    rx: Optional[str] = None,
    yacc: Optional[str] = None,
    stop_at: Optional[str] = None,
    max_tokens=None,
    max_words=None,
    max_bytes=None,
):
    return {
        "Gen": {
            "rx": rx,
            "yacc": yacc,
            "stop_at": stop_at,
            "max_tokens": max_tokens,
            "max_words": max_words,
            "max_bytes": max_bytes,
        }
    }


def fixed(text: str):
    return {"Fixed": {"text": text}}


def choose(options: list[str]):
    return {"Choose": {"options": options}}


def is_step(d: dict):
    return len(d) == 1 and ("Fixed" in d or "Gen" in d or "Choose" in d)


strrx = r'(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)+'


def json_to_steps(json_value):
    # currently we fail for possibly empty rx, so put + not * at the end
    steps = []

    def value_step(v):
        if isinstance(v, bool):
            return choose(["true", "false"])
        elif isinstance(v, int):
            return gen(rx=r"\d{1,10}")
        elif isinstance(v, float):
            return gen(rx=r"\d{1,10}(\.\d{1,10})?")
        elif isinstance(v, str):
            if v == "":
                return gen(rx=strrx, max_words=20)
            elif re.search(r"[\[\.\\{()}*+]", v):
                return gen(rx=f"({v})")
            else:
                return choose(v.split("|"))
        elif v is None:
            return fixed("null")

    def inner(v):
        nonlocal steps
        if isinstance(v, list):
            steps.append(fixed("["))
            if len(v) > 0:
                inner(v[0])
            steps.append(fixed("]"))
        elif isinstance(v, dict):
            if is_step(v):
                steps.append(v)
                return
            steps.append(fixed("{\n"))
            idx = 0
            for k, v in v.items():
                if idx > 0:
                    steps.append(fixed(",\n"))
                idx += 1
                steps.append(fixed(f'"{k}":'))
                if isinstance(v, str):
                    steps += [fixed('"'), value_step(v), fixed('"')]
                else:
                    inner(v)
            steps.append(fixed("\n}"))
        else:
            steps.append(value_step(v))

    inner(json_value)

    new_steps = []
    for step in steps:
        if "Fixed" in step:
            if len(new_steps) > 0 and "Fixed" in new_steps[-1]:
                new_steps[-1]["Fixed"]["text"] += step["Fixed"]["text"]
                continue
        new_steps.append(step)
    return new_steps


def upload_wasm():
    r = subprocess.run(["sh", "wasm.sh", "build"], cwd=prog)
    if r.returncode != 0:
        sys.exit(1)
    file_path = prog + "/target/strip.wasm"
    print("upload module... ", end="")
    with open(file_path, "rb") as f:
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


def ask_completion(
    prompt, aici_module, aici_arg, temperature=0, max_tokens=200, n=1, log=False
):
    json = {
        "model": "",
        "prompt": prompt,
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
                            if l:
                                print(l)
                            if "Previous WASM Error" in l:
                                print("Bailing out due to WASM error")
                                sys.exit(1)
                            # print(f"*** TOK: '{ch['text']}'")
                        else:
                            print(ch["text"], end="")
                    texts[idx] += ch["text"]
            elif decoded_line == "data: [DONE]":
                print(" [DONE]")
            else:
                print(decoded_line)

    for text in texts:
        print("***")
        print(text)
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/response.json"
    with open(path, "w") as f:
        ujson.dump(
            {"request": json, "texts": texts, "response": full_resp}, f, indent=1
        )
    print(f"response saved to {path}")


sys_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. You are concise.
"""


def llama_prompt(prompt):
    return f"[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n [/INST]</s>\n<s>[INST] {prompt} [/INST]\n"


def codellama_prompt(prompt):
    return f"[INST] {prompt} [/INST]\n"


def main():
    # ast = {
    #    "steps": json_to_steps(
    #        {
    #            "name": "",
    #            "valid": True,
    #            "description": "",
    #            "type": "foo|bar|baz|something|else",
    #            "address": {"street": "", "city": "", "state": "[A-Z][A-Z]"},
    #            "age": 1,
    #            "fraction": 1.5,
    #        }
    #    )
    # }

    ast = {
        "steps": [
            gen(
                yacc=open("grammars/c.y").read(),
                # rx="#include(.|\n)*",
                stop_at="\n}",
                max_tokens=100,
            )
        ]
    }
    ast = {
        "steps": [
            {"Fixed": {"text": "I am about "}},
            {"Gen": {"max_tokens": 10, "rx": r"\d+"}},
            {"Fixed": {"text": " years and "}},
            {"Gen": {"max_tokens": 10, "rx": r"\d+"}},
            {"Fixed": {"text": " months."}},
        ]
    }
    mod = upload_wasm()
    ask_completion(
        prompt=codellama_prompt("Write fib function in C"),
        # prompt=llama_prompt("Write fib function in C, respond in code only"),
        aici_module=mod,
        aici_arg=ast,
        n=10,
        temperature=0.5,
        log=True,
        max_tokens=1000,
    )


main()
