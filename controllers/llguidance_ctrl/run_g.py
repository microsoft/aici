import pyaici.rest
import pyaici.cli
import base64
import ujson as json
import binascii
import os

import guidance
from guidance import (
    one_or_more,
    select,
    zero_or_more,
    byte_range,
    char_set,
    capture,
    gen,
    substring,
    optional,
    string,
    lexeme,
    greedy_grammar,
    lazy_grammar,
    commit_point,
)


def main():

    @guidance(stateless=True, dedent=True)
    def character_maker2(lm, id, description, valid_weapons):
        lm += f"""\
        {{
            "name": "{gen('name', stop='"')}",
            "age": {gen('age', regex='[0-9]+', stop=',')},
            "armor": "{select(options=['leather', 'chainmail', 'plate'], name='armor')}",
            "weapon": "{select(options=valid_weapons, name='weapon')}",
            "class": "{gen('class', stop='"')}",
            "mantra": "{gen('mantra', stop='"')}",
            "strength": {gen('strength', regex='[0-9]+', stop=',')},
            "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
        }}"""
        return lm

    grm = "Write a number: " + gen("text", max_tokens=3)
    grm = "Q: 1000 + 3\nA: " + gen("text", regex="[0-9]+", max_tokens=20)
    grm = "Q: 1000 + 3\nA: " + gen("text", regex="[0-9]+", max_tokens=2)

    grm = "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",")

    grm = "this is a test" + gen("test", max_tokens=1)
    grm = (
        "How much is 2 + 2? "
        + gen(name="test", max_tokens=4)
        + gen(name="test2", max_tokens=4)
        + "\n"
    )
    grm = (
        "one, two, three, " + gen(name="a", max_tokens=2) + gen(name="b", max_tokens=2)
    )
    grm = (
        "one, two, three, " + gen(name="a", max_tokens=1) + gen(name="b", max_tokens=1)
    )
    grm = "one, two, three, " + gen(name="a", max_tokens=100)

    prompt = "1. Here is a sentence "
    grm = gen(name="bla", list_append=True, suffix="\n")

    prompt = "Count to 10: 1, 2, 3, 4, 5, 6, 7, "
    grm = gen("text", stop=",")

    prompt = "<color>red</color>\n<color>"
    grm = gen(stop="</color>") + " and test2"

    prompt = ""
    grm = string("this is a test")

    prompt = "How much is 2 + 2? "
    grm = gen(name="test", max_tokens=30, regex=r"[0-9]+", stop=".")

    prompt = ""
    grm = (
        "Name: "
        + gen(
            "name",
            regex="E[a-z]+",
            stop_regex=["[a-b]", "[x-z]"],
            save_stop_text="saved_name_stop",
        )
        + "\nName: "
        + gen(
            "name2",
            regex="E[a-z]+",
            stop_regex=["[a-b]", "[x-z]"],
            save_stop_text="saved_name_stop2",
        )
    )

    grm = character_maker2(1, "A nimble fighter", ["axe", "sword", "bow"])
    prompt = ""

    prompt = "Three things about J. Random Hacker:\n"
    grm = commit_point(
        '"' + byte_range(b"A", b"Z") + one_or_more(byte_range(b"a", b"z")) + '"'
    )

    prompt = ""
    grm = "This is a" + select(name="text", options=["", "nope"])

    # grm = "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=20) + "\n"

    prompt = "a,b,c,"
    grm = one_or_more(byte_range(b"a", b"g") + ",")

    prompt = ""
    grm = one_or_more(byte_range(b"b", b"f"))

    prompt = ""
    grm = guidance.json(schema={"type": "null"})

    # assert grm.match("null")

    grm = guidance.json(
        "OBJ",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"age": {"type": "integer"}},
        },
    )
    # assert grm.match('{"a": 1}')
    prompt = ""
    grm = "Here's some JSON:\n" + grm  # + "\nAnd some more:\n" + grm

    prompt = ""
    grm = optional("A")

    grm = one_or_more(gen(regex="[a-z]"))
    grm = "A odd number is " + gen(
        "number", regex="[0-9]+", max_tokens=5, temperature=0
    )

    grm = (
        "Q: Are dolphins fish?\nA: "
        + gen("dolphins", regex="Yes|No", max_tokens=10)
        + "\nQ: Are sharks fish?\nA: "
        + gen("sharks", regex="Yes|No", max_tokens=10)
    )

    grm = (
        "Power frequency is "
        + gen("number", regex="[0-9]+", max_tokens=5, temperature=0)
        + "Hz; voltage is "
        + gen("number", regex="[0-9]+", max_tokens=5, temperature=0)
        + "V"
    )

    grm = "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=5)

    grm = character_maker2(1, "A nimble fighter", ["axe", "sword", "bow"])

    grm = (
        "Dolphin name: "
        + commit_point(
            '"' + byte_range(b"A", b"Z") + one_or_more(byte_range(b"a", b"z")) + '"'
        )
        + ","
    )

    grm = "Count to 10: 1, 2, 3, 4, 5, 6, 7, " + gen("text", stop=",") + "\nNot quite."

    grm = (
        "Name: "
        + gen(
            "name",
            regex="E[a-z]+",
            stop_regex=["[a-b]", "[x-z]"],
            save_stop_text="saved_name_stop",
        )
        + "\nName: "
        + gen(
            "name2",
            regex="E[a-z]+",
            stop_regex=["[a-b]", "[x-z]"],
            save_stop_text="saved_name_stop2",
        )
    )

    grm = "6 * 7 = " + greedy_grammar(
        body = lexeme("[0-9]{1,3}")
    ) + "\n"
    # assert grm.match("6 * 7 = 42\n")

    grm = (
        "Dolphin name: "
        + commit_point(
            '"' + byte_range(b"A", b"Z") + one_or_more(byte_range(b"a", b"z")) + '"'
        )
        + ","
    )

    grm = gen(regex="a*")
    grm = "6 * 7 = " + gen(regex="5*") + gen(regex="[1-4][0-9]") + "\n"

    grm = "6 * 7 = " + gen("name", max_tokens=2)

    grm = "Name: " + gen('name', max_tokens=2) + " Height: " + gen('height', max_tokens=3)
    grm = "Name: " + gen('name', max_tokens=2) + "Emily Carter is great; Height: " + gen('height', max_tokens=3)

    grm = "123" + gen(name="numbers", regex=r"\d*233", max_tokens=5)

    grm = greedy_grammar(body=lexeme("[0-9]+"),skip_regex=r"\s*") + "x"

    grm = "Here: 2 + 2 = " + guidance.json(name="num", schema={"type": "integer"})
    # grm = guidance.json(name="num", schema={"type": "integer"})
    # m = grm.match("123<s>")
    # print(m)
    # assert m["num"] == "123"

    # grm = "Name: " + gen('name', max_tokens=2) + " Height: " + gen('height', max_tokens=3)



    # g = zero_or_more("a") + "b"
    # assert g.match("b")
    # assert g.match("ab")

    # lm = guidance.models.Mock(b"<s>1234233234<s>")
    # grammar = one_or_more(select(["1", "2"]))
    # lm += grammar

    # grm = greedy_grammar(
    #     body = lexeme("[0-9]+")
    # )

    max_tokens = 7

    serialized = grm.ll_serialize()

    # with open("tmp/long_json_grammar_req.json", "r") as f:
    #     # with open("tmp/email_regex_grammar.json", "r") as f:
    #     max_tokens = 1000
    #     serialized = json.load(f)

    x_serialized = {
        "grammars": [
            {
                "greedy_lexer": False,
                "nodes": [
                    {"Join": {"sequence": [1]}},
                    {"Join": {"sequence": [2, 3]}},
                    {"Gen": {"body_rx": 0, "stop_rx": "", "temperature": None}},
                    {"Select": {"among": [4, 5]}},
                    {"Join": {"sequence": [3, 2]}},
                    {"String": {"literal": ""}},
                ],
                "rx_nodes": [{"ByteSet": [0, 0, 0, 134217726, 0, 0, 0, 0]}],
            }
        ]
    }

    serialized["max_tokens"] = max_tokens
    serialized["test_trace"] = True
    llguidance_json = {"grammar": serialized}

    llguidance_arg = json.dumps(llguidance_json, indent=1)
    # save llguidance_arg to file
    with open("tmp/llguidance_arg.json", "w") as f:
        f.write(llguidance_arg)
    print("JSON size:", len(llguidance_arg), "saved to tmp/llguidance_arg.json")
    # print(json.dumps(llguidance_json, indent=2))

    # with open("tmp/long_json_grammar_req.json", "r") as f:
    #     llguidance_arg = f.read()

    # read current script file
    # with open(__file__) as f:
    #     script = f.read()
    # grm = "```python\n" + substring(script[0:1400])

    features = ["logging"]
    if "FAST" in os.environ:
        features = []
    mod_id = pyaici.cli.build_rust(".", features=features)
    if "127.0.0.1" in pyaici.rest.base_url:
        pyaici.rest.tag_module(mod_id, ["llguidance_ctrl-latest", "llguidance"])
    pyaici.rest.log_level = 2
    res = pyaici.rest.run_controller(
        prompt=prompt,
        controller=mod_id,
        controller_arg=llguidance_arg,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])
    print()

    testcase_from_logs(res["logs"][0])

    text = b""
    captures = {}
    for j in res["json_out"][0]:
        if j["object"] == "text":
            text += binascii.unhexlify(j["hex"])
        elif j["object"] == "capture":
            captures[j["name"]] = binascii.unhexlify(j["hex"]).decode(
                "utf-8", errors="replace"
            )
    print("Captures:", json.dumps(captures, indent=2))
    print("Final text:\n", text.decode("utf-8", errors="replace"))
    print()


def testcase_from_logs(logs: str):
    sep = "‧"
    pairs = []
    prev_res = None
    prompt = None
    for line in logs.split("\n"):
        if line.startswith("TEST: "):
            obj = json.loads(line[6:])
            if prompt is None:
                prompt = obj["res_prompt"]
                continue
            if prev_res:
                pairs.append((prev_res, obj["arg"]))
            prev_res = obj["res"]
    # assert prev_res == "stop"
    testcase = [prompt]
    gen_tokens = []

    def flush_gen_tokens():
        testcase.append(sep.join(gen_tokens))
        gen_tokens.clear()

    for res, arg in pairs:
        print(res, arg)
        if res["sample_mask"]:
            gen_tokens.append(arg["tokens"])
        else:
            splice = res["splices"][0]
            t0 = splice["tokens"]
            assert t0 == arg["tokens"]
            flush_gen_tokens()
            if splice["backtrack"]:
                t0 = str(splice["backtrack"]) + "↶" + t0
            testcase.append(t0)
    if gen_tokens:
        flush_gen_tokens()

    print("Testcase:", testcase)


main()
