from guidance import select, gen
import pyaici.rest
import pyaici.cli
import base64
import ujson as json


import guidance
from guidance import one_or_more, select, zero_or_more

# stateless=True indicates this function does not depend on LLM generations
@guidance(stateless=True)
def number(lm):
    n = one_or_more(select(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
    # Allow for negative or positive numbers
    return lm + select(['-' + n, n])

@guidance(stateless=True)
def operator(lm):
    return lm + select(['+' , '*', '**', '/', '-'])

@guidance(stateless=True)
def expression(lm):
    # Either
    # 1. A number (terminal)
    # 2. two expressions with an operator and optional whitespace
    # 3. An expression with parentheses around it
    return lm + select([
        number(),
        expression() + zero_or_more(' ') +  operator() + zero_or_more(' ') +  expression(),
        '(' + expression() + ')'
    ])

def main():
    grm = (
        "Here's a sample arithmetic expression: " +
        expression() + " = " + number()
    )
    print( base64.b64encode(grm.serialize()).decode("utf-8"))
    mod_id = pyaici.cli.build_rust(".")
    pyaici.rest.log_level = 2
    res = pyaici.rest.run_controller(
        controller=mod_id,
        controller_arg=json.dumps(
            {"guidance_b64": base64.b64encode(grm.serialize()).decode("utf-8")}
        ),
    )
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])
    print("TEXT:", res["text"])


main()
