from guidance import select, gen
import pyaici.rest
import pyaici.cli
import base64
import ujson as json


def main():
    grm = (
        "Here's a "
        + select(["joke", "poem"])
        + " about cats: "
        + gen(regex=r"[A-Z]+")
    )
    mod_id = pyaici.cli.build_rust(".")
    pyaici.rest.run_controller(
        controller=mod_id,
        controller_arg=json.dumps(
            {"guidance_b64": base64.b64encode(grm.serialize()).decode("utf-8")}
        ),
    )


main()
