import pyaici.server as aici


async def test_42():
    await aici.FixedTokens(
        "The ultimate answer to life, the universe and everything is"
    )
    s = await aici.gen_text(regex=r" \d+\.", max_tokens=5, store_var="x")
    aici.check_vars({"x": " 42."})


aici.test(test_42())
