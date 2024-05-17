import pyaici.rest
import pyaici.cli
import base64
import ujson as json


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
    lm += select(['joke', 'poem'], name='type')
    lm += ": "
    lm += gen("words", regex=r"[A-Z ]+", stop="\n")
    grm = lm

    @guidance(stateless=True, dedent=False)
    def rgrammar(lm):
        return lm + "x" + optional(rgrammar())


    grm = select(["1", "12", "123"], name="the number")
    prompt = "<|user|>\nPick a number:\n<|computer|>\n"

    grm = rgrammar()
    prompt = "x"


    # read current script file
    # with open(__file__) as f:
    #     script = f.read()
    # grm = "```python\n" + substring(script[0:1400])
    b64 = base64.b64encode(grm.serialize()).decode("utf-8")
    print(len(b64))
    mod_id = pyaici.cli.build_rust(".")
    if "127.0.0.1" in pyaici.rest.base_url:
        pyaici.rest.tag_module(mod_id, ["guidance_ctrl-latest", "guidance"])
    pyaici.rest.log_level = 2
    res = pyaici.rest.run_controller(
        prompt=prompt,
        controller=mod_id,
        controller_arg=json.dumps({"guidance_b64": b64}),
        temperature=0.0,
        max_tokens=100,
    )
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])
    print("TEXT:", res["text"])


main()
