from guidance import select, gen
import pyaici.rest
import pyaici.cli
import base64
import ujson as json


def main():
    grm = (
        "Here's one-liner "
        + select(["joke", "poem"])
        + " about cats: "
        + gen(stop="\n")
        + "\nScore (out of 10): "
        + gen(regex=r"\d+\.\d+")
    )
    mod_id = pyaici.cli.build_rust(".")
    pyaici.rest.log_level = 2
    res = pyaici.rest.run_controller(
        controller=mod_id,
        controller_arg=json.dumps(
            {"guidance_b64": base64.b64encode(grm.serialize()).decode("utf-8")}
        ),
    )
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])
    print("TEXT:", res["text"])


main()
