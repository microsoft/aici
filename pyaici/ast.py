import re
from typing import Optional, List, Union


def stop():
    """
    Stop generating output.
    """
    return {"Stop": {}}


def e_current():
    return {"Current": {}}


def e_concat(*parts: dict):
    return {"Concat": {"parts": list(parts)}}


def e_list(*parts: dict):
    return {"Concat": {"list": True, "parts": list(parts)}}


def e_str(s: str):
    return {"String": {"str": s}}


def e_var(name: str):
    return {"Var": {"var": name}}


def e_ifeq(a: dict, b: dict, eq: dict, neq: dict):
    return {"IfEq": {"a": a, "b": b, "eq": eq, "neq": neq}}


def e_extract_one(rx: str, src: dict, template: str = "$1"):
    return {"Extract": {"from": src, "rx": rx, "template": template, "list": False}}


def e_extract_all(rx: str, src: dict, template: str = "$1"):
    return {"Extract": {"from": src, "rx": rx, "template": template, "list": True}}


def stmt_set(var: str, expr: dict):
    """
    Set a variable to a value.
    """
    return {"Set": {"var": var, "expr": expr}}


def gen(
    *,
    rx: Optional[str] = None,
    yacc: Optional[str] = None,
    inner: Optional[dict] = None,
    stop_at: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_words: Optional[int] = None,
    max_bytes: Optional[int] = None,
    mask_tags: Optional[List[str]] = None,
    stmts: Optional[List[dict]] = None,
    append_to_var: Optional[str] = None,
    set_var: Optional[str] = None,
    set: Optional[dict] = None,
):
    """
    Generate output with given constraints.
    `rx` is a regular expression to match. If `yacc` is given, it is a yacc grammar to parse.
    `stop_at` is a string to stop at.
    If `max_tokens` is given, stop after that many tokens; similarly for `max_words` and `max_bytes`.
    """
    if not stmts:
        stmts = []
    if set_var is not None:
        stmts.append(stmt_set(set_var, e_current()))
    if append_to_var is not None:
        stmts.append(stmt_set(append_to_var, e_concat(e_var(append_to_var), e_current())))
    if set is not None:
        for var, expr in set.items():
            stmts.append(stmt_set(var, expr))
    if inner is not None:
        inner = [{"after": k, "options": v} for k, v in inner.items()]
    else:
        inner = []

    return {
        "Gen": {
            "rx": rx,
            "yacc": yacc,
            "inner": inner,
            "stop_at": stop_at,
            "max_tokens": max_tokens,
            "max_words": max_words,
            "max_bytes": max_bytes,
            "mask_tags": mask_tags,
            "stmts": stmts,
        }
    }


def fork(*branches: list[dict]):
    return {
        "Fork": {
            "branches": list(branches),
        }
    }


def wait_vars(*vars: str):
    """
    Wait until all variables are set.
    """
    return {"Wait": {"vars": list(vars)}}


def compile_pattern(text: str):
    parts = []
    start = 0

    while True:
        open_brace = text.find("{{", start)
        if open_brace == -1:
            # Add the last part of the text if any
            if start < len(text):
                parts.append(e_str(text[start:]))
            break

        # Add the text before '{{' if any
        if open_brace > start:
            parts.append(e_str(text[start:open_brace]))

        # Find the next '}}'
        close_brace = text.find("}}", open_brace)
        if close_brace == -1:
            # If no closing '}}' found, break the loop
            break

        # Extract the text inside '{{' and '}}'
        parts.append(e_var(text[open_brace + 2 : close_brace]))

        # Update the start position for the next search
        start = close_brace + 2

    return e_concat(*parts)


def fixed(
    text: Union[str, dict],
    expand_vars=False,
    following: Optional[str] = None,
    tag: Optional[str] = None,
):
    """
    Generate fixed text. Same as `choose([text])`.
    """
    if isinstance(text, str):
        if expand_vars:
            text = compile_pattern(text)
        else:
            text = e_str(text)
    return {
        "Fixed": {
            "text": text,
            "tag": tag,
            "following": following,
        }
    }


def label(label: str, step: dict) -> dict:
    k = list(step.keys())[0]
    step[k]["label"] = label
    return step


def choose(options: Union[dict, list[str]]):
    """
    Constrain output to one of the options.
    """
    if isinstance(options, list):
        options = e_list(*[e_str(o) for o in options])
    return {"Choose": {"options": options}}


def is_step(d: dict):
    return len(d) == 1 and (
        "Fixed" in d or "Gen" in d or "Choose" in d or "Fork" in d or "Wait" in d
    )


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
                new_steps[-1]["Fixed"]["text"]["String"]["str"] += step["Fixed"]["text"]["String"]["str"]
                continue
        new_steps.append(step)
    return new_steps


def clear_none(obj):
    if isinstance(obj, dict):
        kk = list(obj.keys())
        for key in kk:
            if obj[key] is None:
                del obj[key]
            else:
                clear_none(obj[key])
    elif isinstance(obj, list):
        for o in obj:
            clear_none(o)