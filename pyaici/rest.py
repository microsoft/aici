import requests
import ujson
from typing import Optional

base_url = "http://127.0.0.1:8080/v1/"
log_level = 1
ast_module = ""


def response_error(kind: str, resp: requests.Response):
    text = resp.text
    try:
        d = ujson.decode(text)
        if "message" in d:
            text = d["message"]
    except:
        pass
    return RuntimeError(
        f"bad response to {kind} {resp.status_code} {resp.reason}: {text}"
    )


def upload_module(file_path: str) -> str:
    """
    Upload a WASM module to the server.
    Returns the module ID.
    """
    if log_level > 0:
        print("upload module... ", end="")
    with open(file_path, "rb") as f:
        resp = requests.post(base_url + "aici_modules", data=f)
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


def completion(
    prompt,
    aici_module=None,
    aici_arg="",
    temperature=0.0,
    max_tokens=200,
    n=1,
    ignore_eos=False,
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
        "ignore_eos": ignore_eos,
    }
    resp = requests.post(base_url + "completions", json=json, stream=True)
    if resp.status_code != 200:
        raise response_error("completions", resp)
    texts = [""] * n
    logs = [""] * n
    full_resp = []
    storage = {}
    res = {
        "request": json,
        "response": full_resp,
        "text": texts,
        "logs": logs,
        "raw_storage": storage,
        "error": None,
    }

    for line in resp.iter_lines():
        if res["error"]:
            break
        if not line:
            continue
        decoded_line: str = line.decode("utf-8")
        if decoded_line.startswith("data: {"):
            d = ujson.decode(decoded_line[6:])
            full_resp.append(d)
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
                if idx == 0:
                    if log_level > 1:
                        l = ch["logs"].rstrip("\n")
                        if l:
                            print(l)
                        # print(f"*** TOK: '{ch['text']}'")
                    elif log_level > 0:
                        print(ch["text"], end="")
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
