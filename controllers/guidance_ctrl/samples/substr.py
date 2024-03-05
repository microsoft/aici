import pyaici.server as aici

earth = """
Earth is rounded into an ellipsoid with a circumference of about 40,000 km. It is the densest planet in the Solar System. Of the four rocky planets, it is the largest and most massive. Earth is about eight light-minutes away from the Sun and orbits it, taking a year (about 365.25 days) to complete one revolution. Earth rotates around its own axis in slightly less than a day (in about 23 hours and 56 minutes). Earth's axis of rotation is tilted with respect to the perPendicular to its orbital plane around the Sun, producing seasons. Earth is orbited by one permanent natural satellite, the Moon, which orbits Earth at 384,400 km (1.28 light seconds) and is roughly a quarter as wide as Earth. Through tidal locking, the Moon always faces Earth with the same side, which causes tides, stabilizes Earth's axis, and gradually slows its rotation.
"""

prompt = f"""[INST] Here's some text:
{earth}

Based on the text answer the question: Is Earth axis aligned with its orbit?
Provide a quote from the text, prefixed by 'Source: "', to support your answer.
[/INST]
"""

async def test_substr():
    await aici.FixedTokens(prompt)
    await aici.gen_tokens(max_tokens=60, stop_at="Source: \"", store_var="answer")
    await aici.gen_tokens(substring=earth, substring_end="\"", max_tokens=60, store_var="source")
    # make sure we can continue generating afterwards
    await aici.FixedTokens("\nThe tilt is")
    await aici.gen_tokens(max_tokens=6, store_var="tilt")

aici.test(test_substr())