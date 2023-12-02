import aici

async def main():
    print("start")
    q = aici.RegexConstraint(r' "[^"]+"')
    await aici.FixedTokens("The word 'hello' in French is")
    await aici.gen_tokens(q)

aici.aici_start(main())
