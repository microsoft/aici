import re
from typing import Optional, List


def gen(
    *,
    rx: Optional[str] = None,
    yacc: Optional[str] = None,
    stop_at: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_words: Optional[int] = None,
    max_bytes: Optional[int] = None,
    mask_tags: Optional[List[str]] = None,
):
    """
    Generate output with given constraints.
    `rx` is a regular expression to match. If `yacc` is given, it is a yacc grammar to parse.
    `stop_at` is a string to stop at.
    If `max_tokens` is given, stop after that many tokens; similarly for `max_words` and `max_bytes`.
    """
    return {
        "Gen": {
            "rx": rx,
            "yacc": yacc,
            "stop_at": stop_at,
            "max_tokens": max_tokens,
            "max_words": max_words,
            "max_bytes": max_bytes,
            "mask_tags": mask_tags,
        }
    }


def fixed(text: str, tag: Optional[str] = None):
    """
    Generate fixed text. Same as `choose([text])`.
    """
    return {"Fixed": {"text": text, "tag": tag}}


def choose(options: list[str]):
    """
    Constrain output to one of the options.
    """
    return {"Choose": {"options": options}}


def is_step(d: dict):
    return len(d) == 1 and ("Fixed" in d or "Gen" in d or "Choose" in d)


# currently we fail for possibly empty rx, so put + not * at the end
strrx = r'(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)+'


def json_to_steps(json_value):
    """
    Given a JSON "schema", generate a sequence of steps that will constrain output to that schema.

    Example:
    >>> json_to_steps({
    ...     "name": "",        # any string will do
    ...     "valid": True,     # True or False
    ...     "description": "",
    ...     # one of these options
    ...     "type": "foo|bar|baz|something|else",
    ...     "address": {
    ...         "street": "",
    ...         "city": "",
    ...         "state": "[A-Z][A-Z]" # regular expression
    ...     },
    ...     "age": 1,        # any integer
    ...     "fraction": 1.5, # any float or integer
    ... })
    """
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
