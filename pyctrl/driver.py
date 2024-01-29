import sys
import os
import re

import pyaici.rest
import pyaici.util
import pyaici.cli


class Tester:
    def __init__(self):
        self.mod = pyaici.cli.build_rust(".")
        self.oks = []
        self.failures = []
        pyaici.rest.log_level = 0

    def fail(self, id: str, logs: str):
        print(f"FAIL")
        self.failures.append((id, logs))

    def ok(self, id: str, logs: str):
        print(f"OK")
        self.oks.append((id, logs))

    def test_one(self, id: str, arg: str):
        try:
            print(f"{id}... ", end="")
            sys.stdout.flush()

            r = pyaici.rest.run_controller(
                controller=self.mod,
                controller_arg=arg,
                max_tokens=2000,
            )
            logs = "\n".join(r["logs"])
            if (
                "\nTEST OK\n" in logs
                and not "\nPanicked at" in logs
                and not "wasm `unreachable`" in logs
                and not "aici_host_stop" in logs
            ):
                self.ok(id, logs)
            else:
                self.fail(id, logs)
        except Exception as e:
            self.fail(id, str(e))


def main():
    os.makedirs("tmp", exist_ok=True)
    js_mode = False
    cmt = "#"

    files = sys.argv[1:]
    if not files:
        print("need some python files as input")
        return

    if files[0].endswith(".js"):
        js_mode = True
        cmt = "//"

    tester = Tester()

    for f in files:
        arg = open(f).read()
        # remove direct calls to tests
        arg = re.subn(
            r"^aici.(start|test)\(.*", cmt + r" \g<0>", arg, flags=re.MULTILINE
        )[0]
        arg = re.subn(r"^(start|test)\(.*", cmt + r" \g<0>", arg, flags=re.MULTILINE)[0]
        # find tests
        if js_mode:
            tests = re.findall(
                r"^async function (test[A-Z_]\w*)\(.*", arg, flags=re.MULTILINE
            )
        else:
            tests = re.findall(r"^async def (test_\w+)\(.*", arg, flags=re.MULTILINE)
        for t in tests:
            if js_mode:
                arg_t = f"{arg}\ntest({t});\n"
            else:
                arg_t = f"{arg}\naici.test({t}())\n"
            tester.test_one(f + "::" + t, arg_t)
            if tester.failures:
                break

    if tester.failures:
        print("\n\nFAILURES:")
        for id, logs in tester.failures:
            print(f"\n*** {id}\n{logs}\n\n")
        sys.exit(1)
    else:
        print(f"All {len(tester.oks)} tests OK")


main()
