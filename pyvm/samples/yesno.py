import aici

async def main():
    tokens = await aici.GetPrompt()
    assert len(tokens) > 2, "prompt too short"
    await aici.FixedTokens("\n")
    await aici.gen_tokens(options=["Yes", "No"])

aici.start(main())
