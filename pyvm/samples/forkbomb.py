import pyaici.server as aici


async def fork_bomb():
    await aici.FixedTokens("The value of")
    id = await aici.fork(20)
    await aici.FixedTokens(f" {id} is")
    await aici.gen_text(max_tokens=5, store_var=f"x{id}")


async def deadlock():
    await aici.wait_vars("foo")


async def burn_tokens():
    id = await aici.fork(10)
    await aici.FixedTokens(f"The value of {id} in the universe is and everything is ")
    await aici.gen_text(max_tokens=200, store_var=f"foo{id}")

aici.test(burn_tokens())
