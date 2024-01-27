import pyaici.server as aici

async def main():
    await aici.FixedTokens("Are dolphins fish?\n")
    await aici.gen_tokens(options=["Yes", "No"])

aici.start(main())
