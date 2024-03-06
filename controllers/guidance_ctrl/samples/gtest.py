import pyaici.gserver as aici


def single_byte(v: int):
    r = aici.ByteSet()
    r.add(v)
    return r


def byte_range(start: int, end: int):
    r = aici.ByteSet()
    r.add_range(start, end)
    return r


def terminal(value: str):
    return [single_byte(b) for b in value.encode("utf-8")]


async def test_joke():
    g = aici.Grammar()
    g.add_rule("expr", ["id"])
    g.add_rule("expr", terminal("(") + ["expr"] + terminal(")"))
    g.add_rule("expr", ["expr", "op", "expr"])
    g.add_rule("op", terminal(" + "))
    g.add_rule("op", terminal(" - "))
    letters = byte_range(ord("a"), ord("z"))
    letters.add(ord("_"))
    # letters.add_range(ord("A"), ord("Z"))
    idchars = byte_range(ord("0"), ord("9"))
    idchars.add_set(letters)
    g.add_rule("id", [letters, "cont"])
    g.add_rule("cont", [])
    g.add_rule("cont", ["cont", idchars])
    g.add_rule("_start", terminal("expression: ") + ["expr"])
    print(repr(g))
    print(repr(g.optimize()))

    await aici.FixedTokens("Hello, ")
    await aici.gen_tokens(constraint=lambda: g.parser(), max_tokens=20)


aici.test(test_joke())
