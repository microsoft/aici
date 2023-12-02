import aici


async def main():
    # init
    print("start")
    print(aici.get_var("test"))
    aici.set_var("test", "hello")
    v=aici.get_var("test")
    print(type(v))
    prompt = await aici.GetPrompt()
    print(prompt)
    await aici.FixedTokens("The word 'hello' in French is")
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="french")
    await aici.FixedTokens("\nAnd in German")
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="german")


aici.aici_start(main())
