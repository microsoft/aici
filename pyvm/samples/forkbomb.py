import aici


async def fork_bomb():
    await aici.FixedTokens("The value of")
    id = await aici.fork(20)
    await aici.FixedTokens(f" {id} is")
    await aici.gen_text(max_tokens=5, store_var=f"x{id}")


aici.test(fork_bomb())
