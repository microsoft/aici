import requests
import json
import os
import urllib.parse
import sys
import time
from typing import Optional

BASE_URL_ENV = "AICI_API_BASE"

base_url = os.environ.get(BASE_URL_ENV, "http://127.0.0.1:8080/v1/")
log_level = 1
ast_module = ""


def require_explicit_base_url():
    if not os.environ.get(BASE_URL_ENV, ""):
        print(f"Please set the {BASE_URL_ENV} environment variable")
        sys.exit(1)


def _parse_base_url(base_url: str):
    p = urllib.parse.urlparse(base_url)
    key = ""
    if p.fragment:
        f = urllib.parse.parse_qs(p.fragment)
        key = f.get("key", [""])[0]
    r = urllib.parse.urlunparse(p._replace(fragment="", query=""))
    if not r.endswith("/"):
        r += "/"
    return r, key


def _headers() -> dict:
    _, key = _parse_base_url(base_url)
    if key:
        return {"api-key": key}
    else:
        return {}


def _mk_url(path: str) -> str:
    pref, _ = _parse_base_url(base_url)
    return pref + path


def response_error(kind: str, resp: requests.Response):
    text = resp.text
    try:
        d = json.loads(text)
        if "message" in d:
            text = d["message"]
    except:
        pass
    return RuntimeError(
        f"bad response to {kind} {resp.status_code} {resp.reason}: {text}"
    )


def req(tp: str, url: str, **kwargs):
    url = _mk_url(url)
    headers = _headers()
    if log_level >= 4:
        print(f"{tp.upper()} {url} headers={headers}")
        if "json" in kwargs:
            print(json.dumps(kwargs["json"]))
    resp = requests.request(tp, url, headers=headers, **kwargs)
    if log_level >= 4 and "stream" not in kwargs:
        print(f"{resp.status_code} {resp.reason}: {resp.text}")
    return resp


def upload_module(file_path: str) -> str:
    """
    Upload a WASM module to the server.
    Returns the module ID.
    """
    if log_level > 0:
        print("upload module... ", end="")
    with open(file_path, "rb") as f:
        resp = req("post", "aici_modules", data=f)
        if resp.status_code == 200:
            dd = resp.json()
            mod_id = dd["module_id"]
            if log_level > 0:
                print(
                    f"{dd['wasm_size']//1024}kB -> {dd['compiled_size']//1024}kB id:{mod_id[0:8]}"
                )
            return mod_id
        else:
            raise response_error("module upload", resp)


def pp_tag(d: dict) -> str:
    t = time.strftime("%F %T", time.localtime(d["updated_at"]))
    M = 1024 * 1024
    return f'{d["tag"]} -> {d["module_id"][0:8]}...; {d["wasm_size"]/M:.3}MiB/{d["compiled_size"]/M:.3}MiB ({t} by {d["updated_by"]})'


def list_tags():
    resp = req("get", "aici_modules/tags")
    if resp.status_code == 200:
        dd = resp.json()
        return dd["tags"]
    else:
        raise response_error("module tag", resp)


def tag_module(module_id: str, tags: list[str]):
    resp = req("post", "aici_modules/tags", json={"module_id": module_id, "tags": tags})
    if resp.status_code == 200:
        dd = resp.json()
        if log_level > 0:
            for t in dd["tags"]:
                print("TAG: " + pp_tag(t))
        return dd["tags"]
    else:
        raise response_error("module tag", resp)


def completion(
    prompt,
    aici_module=None,
    aici_arg="",
    temperature=0.0,
    max_tokens=200,
    n=1,
    ignore_eos: bool | None = None,
):
    if ignore_eos is None:
        ignore_eos = not not ast_module
    data = {
        "model": "",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "stream": True,
        "aici_module": aici_module,
        "aici_arg": aici_arg,
        "ignore_eos": ignore_eos,
    }
    resp = req("post", "completions", json=data, stream=True)
    if resp.status_code != 200:
        raise response_error("completions", resp)
    texts = [""] * n
    logs = [""] * n
    full_resp = []
    storage = {}
    res = {
        "request": data,
        "response": full_resp,
        "text": texts,
        "logs": logs,
        "raw_storage": storage,
        "error": None,
        "usage": {},
    }

    for line in resp.iter_lines():
        if res["error"]:
            break
        if not line:
            continue
        decoded_line: str = line.decode("utf-8")
        # print(decoded_line)
        if decoded_line.startswith("data: {"):
            d = json.loads(decoded_line[6:])
            full_resp.append(d)
            if "usage" in d:
                res["usage"] = d["usage"]
            for ch in d["choices"]:
                if "Previous WASM Error" in ch["logs"]:
                    res["error"] = "WASM error"
                idx = ch["index"]
                while len(texts) <= idx:
                    texts.append("")
                    logs.append("")
                for s in ch.get("storage", []):
                    w = s.get("WriteVar", None)
                    if w:
                        storage[w["name"]] = w["value"]
                err = ch.get("error", "")
                if err:
                    res["error"] = err
                    print(f"*** Error in [{idx}]: {err}")
                if log_level > 2:
                    l = ch["logs"].rstrip("\n")
                    if l:
                        for ll in l.split("\n"):
                            print(f"[{idx}]: {ll}")
                elif idx == 0:
                    if log_level > 1:
                        l = ch["logs"].rstrip("\n")
                        if l:
                            print(l)
                        # print(f"*** TOK: '{ch['text']}'")
                    elif log_level > 0:
                        print(ch["text"], end="")
                        sys.stdout.flush()
                logs[idx] += ch["logs"]
                texts[idx] += ch["text"]
        elif decoded_line == "data: [DONE]":
            if log_level > 0:
                print("[DONE]")
        else:
            raise RuntimeError(f"bad response line: {decoded_line}")

    # convert hex bytes in storage to strings
    s = {}
    res["storage"] = s
    for k, v in storage.items():
        s[k] = bytes.fromhex(v).decode("utf-8", errors="ignore")
    return res
