import pyaici.server as aici
import re

# asserts for microsoft/Orca-2-13b


async def test_joke():
    await aici.FixedTokens("Do you want a joke or a poem? A")
    answer = await aici.gen_text(options=[" joke", " poem"])
    if answer == " joke":
        await aici.FixedTokens("\nHere is a one-line joke about cats: ")
    else:
        await aici.FixedTokens("\nHere is a one-line poem about dogs: ")
    await aici.gen_text(regex="[A-Z].*", stop_at="\n", store_var="result")
    print("explaining...")
    await aici.FixedTokens("\nLet me explain it: ")
    await aici.gen_text(max_tokens=15)


aici.test(test_joke())
