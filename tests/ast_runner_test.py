import ujson
import pytest

import pyaici
import pyaici.rest
import pyaici.util
import pyaici.ast as ast

model_name = "codellama/CodeLlama-13b-Instruct-hf"


def wrap(text):
    return pyaici.util.codellama_prompt(text)


def single_query(prompt: str, steps: list):
    ast_module = pyaici.rest.ast_module
    assert ast_module
    res = pyaici.rest.completion(
        prompt=prompt,
        aici_module=ast_module,
        aici_arg={"steps": steps},
        temperature=0,
        max_tokens=200,
        n=1,
    )
    if res["error"]:
        pytest.fail(res["error"])
    return res["text"][0]


def expect(expected: str, prompt: str, steps: list):
    res = single_query(prompt, steps)
    if res != expected:
        if len(res) > 40:
            print(f'"""{res}"""')
        else:
            print(ujson.dumps(res))
        pytest.fail("query output mismatch")


def test_hello():
    expect(
        ", I am",
        "Hello",
        [ast.gen(max_tokens=3)],
    )


def test_gen_num():
    expect(
        "I am about 10 years and 10 months.",
        "",
        [
            ast.fixed("I am about "),
            ast.gen(max_tokens=5, rx=r"\d+"),
            ast.fixed(" years and "),
            ast.gen(max_tokens=5, rx=r"\d+"),
            ast.fixed(" months."),
        ],
    )


def test_grammar():
    expect(
        """```
int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n-1) + fib(n-2);
}""",
        wrap("Write fib function in C"),
        [
            ast.fixed("```\n"),
            ast.gen(
                yacc=open("grammars/c.y").read(),
                # rx="#include(.|\n)*",
                stop_at="\n}",
                max_tokens=100,
            ),
        ],
    )


def test_json():
    expect(
        """{
"name":"J. Random Hacker",
"valid":true,
"description":"J. Random Hacker is a legendary hacker from Seattle, known for his unparalleled skills in computer security and his unwavering dedic",
"type":"bar",
"address":{
"street":"123 Main St",
"city":"Seattle",
"state":"WA"
},
"age":35,
"fraction":0.5
}""",
        wrap("Write about J. Random Hacker from Seattle"),
        ast.json_to_steps(
            {
                "name": "",
                "valid": True,
                "description": "",
                "type": "foo|bar|baz|something|else",
                "address": {"street": "", "city": "", "state": "[A-Z][A-Z]"},
                "age": 1,
                "fraction": 1.5,
            }
        ),
    )


def test_ff_0():
    expect(
        ", 3 + 8 is 11.\n",
        "Hello",
        [
            {"Gen": {"rx": ", ", "max_tokens": 10}},
            {"Fixed": {"text": "3 + 8 is"}},
            {"Gen": {"max_tokens": 5}},
        ],
    )


def test_ff_1():
    expect(
        ", 7 + 8 = 15",
        "Hello",
        [
            ast.gen(rx=", "),
            ast.fixed("7 + 8 ="),
            ast.gen(rx=r" \d+"),
        ],
    )


def test_ff_2():
    expect(
        ", 7 + 8 = 15",
        "Hello",
        [
            ast.gen(rx=", "),
            ast.fixed("7 + 8"),
            ast.gen(rx=r" = \d+"),
        ],
    )


def check_mask(expected, mask_tags):
    expect(
        expected,
        "The word 'hello' in",
        [
            ast.fixed(" French is", tag="lang"),
            ast.gen(max_tokens=5, mask_tags=mask_tags),
        ],
    )

def test_mask_1():
    check_mask(" French is 'hello world' is", ["lang"])

def test_mask_2():
    check_mask(" French is 'bonjour'.", [])
