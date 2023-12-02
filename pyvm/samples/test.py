import aici


async def main():
    print("start")
    await aici.FixedTokens("The word 'hello' in French is")
    await aici.gen_tokens(regex=r' "[^"]+"')


aici.aici_start(main())
