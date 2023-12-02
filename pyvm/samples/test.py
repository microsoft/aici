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
    t = await aici.gen_tokens(regex=r' "[^"]+"')
    print(t)
    s = aici.detokenize(t)
    print(s)


aici.aici_start(main())
