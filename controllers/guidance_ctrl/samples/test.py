import pyaici.server as aici
import re

# asserts for microsoft/Orca-2-13b

async def test_backtrack_one():
    await aici.FixedTokens("3+")
    l = aici.Label()
    await aici.FixedTokens("2")
    await aici.gen_tokens(regex=r"=\d\d?\.", store_var="x", max_tokens=5)
    print("X", aici.get_tokens(), aici.detokenize(aici.get_tokens()))
    await aici.FixedTokens("4", following=l)
    await aici.gen_tokens(regex=r"=\d\d?\.", store_var="y", max_tokens=5)
    print("Y", aici.get_tokens(), aici.detokenize(aici.get_tokens()))
    aici.check_vars({"x": "=5.", "y": "=7."})


async def test_fork():
    await aici.FixedTokens("The word 'hello' in")
    id = await aici.fork(3)
    if id == 0:
        french, german = await aici.wait_vars("french", "german")
        await aici.FixedTokens(f"{french} is the same as {german}.")
        await aici.gen_tokens(max_tokens=5)
        aici.check_vars({"german": ' "hallo"', "french": ' "bonjour"'})
    elif id == 1:
        await aici.FixedTokens(" German is")
        await aici.gen_tokens(regex=r' "[^"\.]+"', store_var="german", max_tokens=5)
    elif id == 2:
        await aici.FixedTokens(" French is")
        await aici.gen_tokens(regex=r' "[^"\.]+"', store_var="french", max_tokens=5)


async def test_backtrack_lang():
    await aici.FixedTokens("The word 'hello' in")
    l = aici.Label()
    await aici.FixedTokens(" French is", following=l)
    await aici.gen_tokens(regex=r' "[^"\.]+"', store_var="french", max_tokens=5)
    await aici.FixedTokens(" German is", following=l)
    await aici.gen_tokens(regex=r' "[^"\.]+"', store_var="german", max_tokens=5)
    aici.check_vars({"french": ' "bonjour"', "german": ' "hallo"'})


async def test_main():
    # init
    print("start")
    print(aici.get_var("test"))
    aici.set_var("test", "hello")
    v = aici.get_var("test")
    print(type(v))
    await aici.FixedTokens("The word 'hello' in French is")
    # try unconstrained output
    await aici.gen_tokens(store_var="french", max_tokens=5)
    await aici.FixedTokens("\nIn German it translates to")
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
    aici.check_vars(
        {
            "test": "hello",
            "french": " 'bonjour'.",
            "german": ' "guten Tag"',
            "five": " pounds",
            "dollars": "7.5",
        }
    )


async def test_drugs():
    drug_syn = "\nUse <drug>Drug Name</drug> syntax for any drug name, for example <drug>Advil</drug>.\n\n"

    notes = "The patient should take some tylenol in the evening and aspirin in the morning. Exercise is highly recommended. Get lots of sleep.\n"
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
        max_tokens=30,
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

    aici.check_vars(
        {
            "times": "<drug>Tylenol</drug> in the evening.\n"
            "Take <drug>Aspirin</drug> in the morning.\n"
            "Exercise is highly recommended.\nGet lots of sleep."
        }
    )


async def test_sample():
    # initialization code
    print("I'm going in the logs!")
    # ... more initialization code, it has long time limit

    # This appends the exact string to the output; similar to adding it to prompt
    await aici.FixedTokens("The word 'hello' in French is")

    # here we're out of initialization code - the time limits are tight

    # generate text (tokens) matching the regex
    french = await aici.gen_text(regex=r' "[^"]+"', max_tokens=5)
    # set a shared variable (they are returned as JSON and are useful with aici.fork())
    aici.set_var("french", french)

    await aici.FixedTokens(" and in German")
    # shorthand for the above
    await aici.gen_text(regex=r' "[^"]+"', store_var="german")

    await aici.FixedTokens("\nFive")
    # generates one of the strings
    await aici.gen_text(options=[" pounds", " euros", " dollars"])


class SampleEos(aici.NextToken):
    def mid_process(self) -> aici.MidProcessResult:
        ts = aici.TokenSet()
        ts[aici.eos_token()] = True
        return aici.MidProcessResult.bias(ts)


async def test_eos():
    await aici.FixedTokens("The word 'hello' in French is")
    await SampleEos()
    await aici.gen_tokens(regex=r' "[^"]+"', max_tokens=6, store_var="french")
    aici.check_vars({"french": ' "bonjour"'})

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
