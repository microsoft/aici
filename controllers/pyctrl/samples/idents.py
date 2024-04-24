import pyaici.server as aici
import re

# asserts for microsoft/Orca-2-13b

aici.log_level = 1

async def test_id():
    await aici.FixedTokens("Here's a fib function\n```python\n")

    max_tokens = 60
    dyn_lex = aici.DynamicLexer("")
    for id in ["def", "fibo", "n", "return", "if"]:
        dyn_lex.add(id)
    next_token = aici.ConstrainedToken(lambda: dyn_lex.constraint())
    res = []
    text = ""
    for _ in range(max_tokens):
        tokens = await next_token
        if tokens:
            res += tokens
            print("GEN-STEP:", aici.tokens_repr(tokens))
            text = aici.detokenize(res).decode(errors="replace")
        if next_token.finished:
            break
    print("RESULT:", text)
    

aici.test(test_id())
