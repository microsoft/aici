import aici

async def main():
    print("start")
    await aici.FixedTokens("The word 'hello' in French")
    await aici.sample_gen_tokens(10)

aici.aici_start(main())
