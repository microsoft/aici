import aici
import re


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


async def main_fork():
    await aici.FixedTokens("The word 'hello' in")
    id = await aici.fork(3)
    if id == 0:
        french, german = await aici.wait_vars("french", "german")
        await aici.FixedTokens(f"{french} is the same as {german}.")
        await aici.gen_tokens(max_tokens=5)
    elif id == 1:
        await aici.FixedTokens(" German is")
        await aici.gen_tokens(regex=r' "[^"]+"', store_var="german", max_tokens=5)
    elif id == 2:
        await aici.FixedTokens(" French is")
        await aici.gen_tokens(regex=r' "[^"]+"', store_var="french", max_tokens=5)
    check_vars({"french": ' "bonjour"', "german": ' "Hallo"'})


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
    prompt = await aici.GetPrompt()
    print(prompt)
    await aici.FixedTokens("The word 'hello' in French is")
    # try unconstrained output
    await aici.gen_tokens(store_var="french", max_tokens=5)
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
    await aici.gen_tokens(regex=r"\d+\.\d", store_var="dollars")
    check_vars(
        {
            "test": "hello",
            "french": " 'bonjour'.",
            "german": ' "Hallo"',
            "five": " pounds",
            "dollars": "100.0",
        }
    )


async def drugs():
    drug_syn = "\nUse <drug>Drug Name</drug> syntax for any drug name, for example <drug>Advil</drug>.\n\n"

    notes = "The patient should take some tylenol in the evening and aspirin in the morning. Excercise is highly recommended. Get lots of sleep.\n"
    notes = "Start doctor note:\n" + notes + "\nEnd doctor note.\n"

    await aici.FixedTokens("[INST] ")
    start = aici.Label()

    def inst(s: str) -> str:
        return s + drug_syn + notes + " [/INST]\n"

    await aici.FixedTokens(
        inst("List specific drug names in the following doctor's notes.")
        + "\n1. <drug>"
    )
    s = await aici.gen_text(
        max_tokens=100,
    )
    drugs = re.findall(r"<drug>([^<]*)</drug>", "<drug>" + s)
    print("drugs", drugs)
    await aici.FixedTokens(
        inst(
            "Make a list of each drug along with time to take it, based on the following doctor's notes."
        )
        + "Take <drug>",
        following=start,
    )
    pos = aici.Label()
    await aici.gen_tokens(options=[d + "</drug>" for d in drugs])
    for _ in range(5):
        fragment = await aici.gen_text(max_tokens=20, stop_at="<drug>")
        print(fragment)
        if "<drug>" in fragment:
            assert fragment.endswith("<drug>")
            await aici.gen_tokens(options=[d + "</drug>" for d in drugs])
        else:
            break

    aici.set_var("times", "<drug>" + pos.text_since())

    check_vars(
        {
            "times": "<drug>Tylenol</drug> in the evening.\n"
            "Take <drug>Aspirin</drug> in the morning.\n"
            "Exercise is highly recommended.\nGet lots of sleep."
        }
    )


aici.aici_start(drugs())
