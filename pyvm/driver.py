import subprocess
import sys
import os
import re

import pyaici.ast as ast
import pyaici.rest
import pyaici.util


def upload_wasm(prog="."):
    r = subprocess.run(["sh", "wasm.sh", "build"], cwd=prog)
    if r.returncode != 0:
        sys.exit(1)
    file_path = "../target/opt.wasm"
    return pyaici.rest.upload_module(file_path)


class Tester:
    def __init__(self):
        self.mod = upload_wasm()
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

            r = pyaici.rest.completion(
                prompt="",
                aici_module=self.mod,
                aici_arg=arg,
                ignore_eos=True,
                max_tokens=2000,
            )
            logs = "\n".join(r["logs"])
            if (
                "\nTEST OK\n" in logs
                and not "\nPanicked at" in logs
                and not "wasm `unreachable`" in logs
            ):
                self.ok(id, logs)
            else:
                self.fail(id, logs)
        except Exception as e:
            self.fail(id, str(e))


def main():
    os.makedirs("tmp", exist_ok=True)

    files = sys.argv[1:]
    if not files:
        print("need some python files as input")
        return

    tester = Tester()

    for f in files:
        arg = open(f).read()
        # remove direct calls to tests
        arg = re.subn(r"^aici.(start|test)\(.*", r"# \g<0>", arg, flags=re.MULTILINE)[0]
        # find tests
        tests = re.findall(r"^async def (test_\w+)\(.*", arg, flags=re.MULTILINE)
        for t in tests:
            arg_t = arg + f"{arg}\naici.test({t}())\n"
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
