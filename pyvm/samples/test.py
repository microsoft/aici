import aici


async def main():
    # init
    print("start")
    print(aici.get_var("test"))
    aici.set_var("test", "hello")
    v = aici.get_var("test")
    print(type(v))
    prompt = await aici.GetPrompt()
    print(prompt)
    await aici.FixedTokens("The word 'hello' in French is")
    await aici.gen_tokens(store_var="french", max_tokens=5)
    # await aici.gen_tokens(regex=r' "[^"]+"', store_var="french")
    await aici.FixedTokens("\nAnd in German")
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="german")
    await aici.FixedTokens("\nFive")
    await aici.gen_tokens(
        store_var="five",
        options=[
            " pounds",
            " euros",
        ],
    )
    await aici.FixedTokens(" is worth about $")
    await aici.gen_tokens(regex=r'\d+\.\d', store_var="dollars")


aici.aici_start(main())
