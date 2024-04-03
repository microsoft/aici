import pyaici.rest
import pyaici.cli
import base64
import ujson as json


import guidance
from guidance import one_or_more, select, zero_or_more, byte_range, capture, gen


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
        "<joke>Parallel lines have so much in common. It’s a shame they’ll never meet.</joke>\n"
        + "<joke>"
        + capture(gen(regex=r'[A-Z\(].*', stop="</joke>"), "joke")
        + "</joke>\nScore (of 10): "
        + capture(gen(regex=r"\d{1,3}"), "score")
        + "\n"
    )
    print(base64.b64encode(grm.serialize()).decode("utf-8"))
    mod_id = pyaici.cli.build_rust(".")
    pyaici.rest.log_level = 2
    res = pyaici.rest.run_controller(
        controller=mod_id,
        controller_arg=json.dumps(
            {"guidance_b64": base64.b64encode(grm.serialize()).decode("utf-8")}
        ),
        max_tokens=100,
    )
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])
    print("TEXT:", res["text"])


main()
