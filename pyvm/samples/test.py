import aici


def check_var(name: str, value: str):
    v = aici.get_var(name)
    if v is None:
        raise Exception(f"ERROR: {name} is None")
    v = v.decode()
    if v != value:
        raise Exception(f"ERROR: {name}={v} != {value}")


def check_vars(d: dict[str, str]):
    for k, v in d.items():
        check_var(k, v)


async def main3():
    await aici.FixedTokens("3+")
    l = aici.Label()
    await aici.FixedTokens("2")
    await aici.gen_tokens(regex=r"=\d\d?\.", store_var="x", max_tokens=5)
    print("X", aici.get_tokens(), aici.detokenize(aici.get_tokens()))
    await aici.FixedTokens("4", following=l)
    await aici.gen_tokens(regex=r"=\d\d?\.", store_var="y", max_tokens=5)
    print("Y", aici.get_tokens(), aici.detokenize(aici.get_tokens()))
    check_vars({"x": "=5.", "y": "=7."})


async def main2():
    await aici.FixedTokens("The word 'hello' in")
    l = aici.Label()
    await aici.FixedTokens(" French is", following=l)
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="french", max_tokens=5)
    await aici.FixedTokens(" German is", following=l)
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="german", max_tokens=5)
    check_vars({"french": ' "bonjour"', "german": ' "Hallo"'})


async def main():
    # init
    print("start")
    print(aici.get_var("test"))
    aici.set_var("test", "hello")
    v = aici.get_var("test")
    print(type(v))
    check_var("test", "hello")
    prompt = await aici.GetPrompt()
    print(prompt)
    await aici.FixedTokens("The word 'hello' in French is")
    # try unconstrained output
    await aici.gen_tokens(store_var="french", max_tokens=5)
    check_var("french", " 'bonjour'.")
    await aici.FixedTokens("\nAnd in German")
    await aici.gen_tokens(regex=r' "[^"]+"', store_var="german")
    check_var("german", ' "Hallo"')
    await aici.FixedTokens("\nFive")
    await aici.gen_tokens(
        store_var="five",
        options=[
            " pounds",
            " euros",
        ],
    )
    check_var("five", " pounds")
    await aici.FixedTokens(" is worth about $")
    await aici.gen_tokens(regex=r"\d+\.\d", store_var="dollars")
    check_var("dollars", "100.0")


aici.aici_start(main3())
