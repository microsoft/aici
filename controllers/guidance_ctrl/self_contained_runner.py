import base64
import ujson as json
import requests
import urllib.parse
import time
import os
import sys
from typing import Optional, Callable

BASE_URL_ENV = "AICI_API_BASE"

base_url = os.environ.get(BASE_URL_ENV, "http://127.0.0.1:4242/v1/")
log_level = 1
ast_module = ""


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
        f"bad response to {kind} {resp.status_code} {resp.reason}: {text}")


def req(tp: str, path: str, base_url: Optional[str] = None, **kwargs):
    url = _mk_url(path, arg_base_url=base_url)
    headers = _headers(arg_base_url=base_url)
    if log_level >= 4:
        print(f"{tp.upper()} {url} headers={headers}")
        if "json" in kwargs:
            print(json.dumps(kwargs["json"]))
    resp = requests.request(tp, url, headers=headers, **kwargs)
    if log_level >= 4 and "stream" not in kwargs:
        print(f"{resp.status_code} {resp.reason}: {resp.text}")
    return resp


def run_controller(
    *,
    controller,
    controller_arg="",
    prompt="",
    json_cb: Optional[Callable[[dict], None]] = None,
    text_cb: Optional[Callable[[str], None]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 200,
    base_url: Optional[str] = None,
):
    data = {
        "controller": controller,
        "controller_arg": controller_arg,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    t0 = time.time()
    resp = req("post", "run", json=data, stream=True, base_url=base_url)
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
        "timing": {
            "http_response": time.time() - t0,
        },
        "tps": {},
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
            if "data0" not in res["timing"]:
                res["timing"]["data0"] = time.time() - t0
            if "forks" not in d:
                continue
            if "first_token" not in res["timing"]:
                res["timing"]["first_token"] = time.time() - t0
                prompt_time = res["timing"]["first_token"] - res["timing"][
                    "http_response"]
                res["tps"]["prompt"] = d["usage"]["ff_tokens"] / prompt_time
            # in guidance, there should be only one fork
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
                if err:
                    res["error"] = err
                    if log_level > 2 and err in ch["logs"]:
                        print(f"*** Error in [{idx}]")
                    else:
                        print(f"*** Error in [{idx}]: {err}")
                if json_cb:
                    for ln in ch["logs"].split("\n"):
                        if ln.startswith("JSON-OUT: "):
                            j = json.loads(ln[10:])
                            json_cb(j)
                if text_cb:
                    text_cb(ch["text"])
                logs[idx] += ch["logs"]
                texts[idx] += ch["text"]
        elif decoded_line == "data: [DONE]":
            if log_level > 0:
                print("[DONE]")
        else:
            raise RuntimeError(f"bad response line: {decoded_line}")

    res["timing"]["last_token"] = time.time() - t0
    res["tps"]["sampling"] = res["usage"]["sampled_tokens"] / res["timing"][
        "last_token"]
    # convert hex bytes in storage to strings
    s = {}
    res["storage"] = s
    for k, v in storage.items():
        s[k] = bytes.fromhex(v).decode("utf-8", errors="ignore")
    return res


#
#
# Testing
#
#

import guidance
from guidance import one_or_more, select, zero_or_more, byte_range, capture, gen, substring


@guidance(stateless=True)
def number(lm):
    n = one_or_more(select(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]))
    return lm + select(["-" + n, n])


@guidance(stateless=True)
def identifier(lm):
    letter = select([byte_range(b"a", b"z"), byte_range(b"A", b"Z"), "_"])
    num = byte_range(b"0", b"9")
    return lm + letter + zero_or_more(select([letter, num]))


@guidance(stateless=True)
def assignment_stmt(lm):
    return lm + identifier() + " = " + expression()


@guidance(stateless=True)
def while_stmt(lm):
    return lm + "while " + expression() + ":" + stmt()


@guidance(stateless=True)
def stmt(lm):
    return lm + select([assignment_stmt(), while_stmt()])


@guidance(stateless=True)
def operator(lm):
    return lm + select(["+", "*", "**", "/", "-"])


@guidance(stateless=True)
def expression(lm):
    return lm + select([
        identifier(),
        expression() + zero_or_more(" ") + operator() + zero_or_more(" ") +
        expression(),
        "(" + expression() + ")",
    ])


def main():
    global log_level
    # Set to "2" to see output from guidance_ctrl
    # Set to more to get info about requests being made
    log_level = 0

    grm = ("<joke>" + capture(
        gen(regex=r'[A-Z\(].*', max_tokens=50, stop="</joke>"), "joke") +
           "</joke>\nScore: " + capture(gen(regex=r"\d{1,3}"), "score") +
           "/10\n")

    # this is called for captures and final text
    def print_json(obj):
        print("JSON:", obj)

    # this callback can be used for progress, but the text returned is "approximate"
    def print_text(txt):
        print("TXT:", repr(txt))

    b64 = base64.b64encode(grm.serialize()).decode("utf-8")
    print("Calling guidance_ctrl; grammar size:", len(b64))

    res = run_controller(
        controller="guidance_ctrl-latest",
        controller_arg=json.dumps({"guidance_b64": b64}),
        prompt=
        "<joke>Parallel lines have so much in common. It’s a shame they’ll never meet.</joke>\nScore: 8/10\n",
        json_cb=print_json,
        text_cb=print_text,
        temperature=0.5,
        max_tokens=100,
    )

    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])


main()
