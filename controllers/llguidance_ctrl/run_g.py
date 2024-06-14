import pyaici.rest
import pyaici.cli
import base64
import ujson as json
import binascii


import guidance
from guidance import (
    one_or_more,
    select,
    zero_or_more,
    byte_range,
    capture,
    gen,
    substring,
    optional,
    string,
    lexeme,
    greedy_grammar,
    lazy_grammar,
)


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
    return lm + select(
        [
            identifier(),
            expression()
            + zero_or_more(" ")
            + operator()
            + zero_or_more(" ")
            + expression(),
            "(" + expression() + ")",
        ]
    )


@guidance(stateless=True)
def json_string(lm):
    return lm + lexeme(r'"(\\(["\\\/bfnrt]|u[a-fA-F0-9]{4})|[^"\\\x00-\x1F\x7F]+)*"')


@guidance(stateless=True)
def json_number(lm):
    return lm + lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


@guidance(stateless=True)
def json_value(lm):
    return lm + select(
        [
            json_string(),
            json_number(),
            json_object(),
            json_array(),
            "true",
            "false",
            "null",
        ]
    )


@guidance(stateless=True)
def json_member(lm):
    return lm + json_string() + ":" + json_value()


@guidance(stateless=True)
def json_object(lm):
    return lm + "{" + optional(json_member() + one_or_more("," + json_member())) + "}"


@guidance(stateless=True)
def json_array(lm):
    return lm + "[" + optional(json_value() + one_or_more("," + json_value())) + "]"


@guidance(stateless=True)
def gen_json_object(lm, name: str, max_tokens=100000000):
    grm = greedy_grammar(body=json_object(), skip_regex=r"[\x20\x0A\x0D\x09]+")
    return lm + grm


def main():
    grm = (
        "Here's a sample arithmetic expression: "
        + capture(expression(), "expr")
        + " = "
        + capture(number(), "num")
    )
    grm = (
        "<joke>Parallel lines have so much in common. It’s a shame they’ll never meet.</joke>\nScore: 8/10\n"
        + "<joke>"
        + capture(gen(regex=r"[A-Z\(].*", max_tokens=50, stop="</joke>"), "joke")
        + "</joke>\nScore: "
        + capture(gen(regex=r"\d{1,3}"), "score")
        + "/10\n"
    )
    grm = "this is a test" + gen("test", max_tokens=10)
    grm = "Tweak this proverb to apply to model instructions instead.\n" + gen(
        "verse", max_tokens=2
    )
    grm = "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\(")
    grm = "<color>red</color>\n<color>" + gen(stop="</color>") + " and test2"

    lm = "Here's a "
    lm += select(["joke", "poem"], name="type")
    lm += ": "
    lm += gen("words", regex=r"[A-Z ]+", stop="\n")
    grm = lm

    @guidance(stateless=True, dedent=True)
    def character_maker(lm, id, description, valid_weapons):
        lm += f"""\
        The following is a character profile for an RPG game in JSON format.
        ```json
        {{
            "id": "{id}",
            "description": "{description}",
            "name": "{gen('name', stop='"')}",
            "age": {gen('age', regex='[0-9]+', stop=',')},
            "armor": "{select(options=['leather', 'chainmail', 'plate'], name='armor')}",
            "weapon": "{select(options=valid_weapons, name='weapon')}",
            "class": "{gen('class', stop='"')}",
            "mantra": "{gen('mantra', stop='"')}",
            "strength": {gen('strength', regex='[0-9]+', stop=',')},
            "items": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
        }}```"""
        return lm

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
            "items_with_colors": ["{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}", "{gen('item', list_append=True, stop='"')}"]
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
    grm = "Name: " + \
        gen('name', regex="E[a-z]+", stop_regex=["[a-b]", "[x-z]"], save_stop_text="saved_name_stop") + \
        "\nName: " + \
        gen('name2', regex="E[a-z]+", stop_regex=["[a-b]", "[x-z]"], save_stop_text="saved_name_stop2")

    prompt = "Three things about J. Random Hacker:\n"
    grm = (
        gen_json_object("hacker", max_tokens=150)
        + "\nScore (0-9): "
        + gen("score", regex=r"[0-9]")
    )

    grm = character_maker2(1, "A nimble fighter", ["axe", "sword", "bow"])
    prompt = ""


    # grm = "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=20) + "\n"

    max_tokens = 250

    serialized = grm.ll_serialize()
    serialized["max_tokens"] = max_tokens
    llguidance_json = {"grammar": serialized}
    llguidance_arg = json.dumps(llguidance_json, indent=1)
    # save llguidance_arg to file
    with open("tmp/llguidance_arg.json", "w") as f:
        f.write(llguidance_arg)
    print("JSON size:", len(llguidance_arg), "saved to tmp/llguidance_arg.json")
    # print(json.dumps(llguidance_json, indent=2))

    # read current script file
    # with open(__file__) as f:
    #     script = f.read()
    # grm = "```python\n" + substring(script[0:1400])

    mod_id = pyaici.cli.build_rust(".", features=["logging"])
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


main()
