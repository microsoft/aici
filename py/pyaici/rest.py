import requests
import json
import os
import urllib.parse
import sys
import time
import re
from typing import Optional, List

BASE_URL_ENV = "AICI_API_BASE"

base_url = os.environ.get(BASE_URL_ENV, "http://127.0.0.1:4242/v1/")
log_level = 1
ast_module = ""


def _clear_none(obj):
    if isinstance(obj, dict):
        kk = list(obj.keys())
        for key in kk:
            if obj[key] is None:
                del obj[key]
            else:
                _clear_none(obj[key])
    elif isinstance(obj, list):
        for o in obj:
            _clear_none(o)
    return obj


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


def _headers(arg_base_url: Optional[str] = None) -> dict:
    if not arg_base_url:
        arg_base_url = base_url
    _, key = _parse_base_url(arg_base_url)
    if key:
        return {"api-key": key}
    else:
        return {}


def _mk_url(path: str, arg_base_url: Optional[str] = None) -> str:
    if not arg_base_url:
        arg_base_url = base_url
    pref, _ = _parse_base_url(arg_base_url)
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


def strip_url_path(url):
    pattern = r"^(https?://[^/]+)"
    match = re.match(pattern, url)
    assert match
    return match.group(1)


def req(tp: str, path: str, base_url: Optional[str] = None, **kwargs):
    url = _mk_url(path, arg_base_url=base_url)
    if path == "/proxy/info":
        url = strip_url_path(url) + path
    headers = _headers(arg_base_url=base_url)
    if log_level >= 4:
        print(f"{tp.upper()} {url} headers={headers}")
        if "json" in kwargs:
            print(json.dumps(kwargs["json"]))
    resp = requests.request(tp, url, headers=headers, **kwargs)
    if log_level >= 4 and "stream" not in kwargs:
        print(f"{resp.status_code} {resp.reason}: {resp.text}")
    return resp


def detect_prefixes():
    resp = req("get", "/proxy/info")
    if resp.status_code == 200:
        dd = resp.json()
        return dd["prefixes"]
    else:
        return ["/"]


def upload_module(file_path: str) -> str:
    """
    Upload a WASM module to the server.
    Returns the module ID.
    """
    if log_level > 0:
        print("upload module... ", end="")
    with open(file_path, "rb") as f:
        resp = req("post", "controllers", data=f)
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
    resp = req("get", "controllers/tags")
    if resp.status_code == 200:
        dd = resp.json()
        return dd["tags"]
    else:
        raise response_error("module tag", resp)


def tag_module(module_id: str, tags: List[str]):
    resp = req("post", "controllers/tags", json={"module_id": module_id, "tags": tags})
    if resp.status_code == 200:
        dd = resp.json()
        if log_level > 0:
            for t in dd["tags"]:
                print("TAG: " + pp_tag(t))
        return dd["tags"]
    else:
        raise response_error("module tag", resp)


def run_controller(
    *,
    controller,
    controller_arg="",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 200,
    base_url: Optional[str] = None,
):
    data = {
        "controller": controller,
        "controller_arg": controller_arg,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = req("post", "run", json=_clear_none(data), stream=True, base_url=base_url)
    if resp.status_code != 200:
        raise response_error("run", resp)
    texts = [""]
    logs = [""]
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
        if log_level >= 5:
            print(decoded_line)
        if decoded_line.startswith("data: {"):
            d = json.loads(decoded_line[6:])
            full_resp.append(d)
            if "usage" in d:
                res["usage"] = d["usage"]
            if "forks" not in d:
                continue
            for ch in d["forks"]:
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
