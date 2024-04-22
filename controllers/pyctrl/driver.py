import sys
import os
import re
import argparse

import pyaici.rest
import pyaici.util
import pyaici.cli


class Tester:
    def __init__(self, controller=None):
        self.mod = controller if controller is not None else pyaici.cli.build_rust(".")
        self.oks = []
        self.failures = []

    def fail(self, id: str, logs: str):
        print("FAIL")
        self.failures.append((id, logs))

    def ok(self, id: str, logs: str):
        print("OK")
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
            nlogs = "\n" + logs
            if (
                "\nTEST OK\n" in nlogs
                and "\nPanicked at" not in nlogs
                and "wasm `unreachable`" not in nlogs
                and "aici_host_stop" not in nlogs
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

    parser = argparse.ArgumentParser(
        description="Run pyctrl or jsctrl tests",
        prog="ctrldriver",
    )

    parser.add_argument(
        "--skip",
        "-s",
        type=str,
        default=[],
        action="append",
        help="skip tests matching string",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=int,
        default=0,
        help="AICI log level",
    )

    parser.add_argument(
        "--only",
        "-k",
        type=str,
        default=[],
        action="append",
        help="only run tests matching string",
    )

    parser.add_argument(
        "--controller",
        "-c",
        type=str,
        help="controller to use (defaults to build current folder)",
    )

    parser.add_argument(
        "test_file",
        nargs="+",
        help="files to test",
    )


    args = parser.parse_args()

    pyaici.rest.log_level = args.log_level

    files = args.test_file

    if files[0].endswith(".js"):
        js_mode = True
        cmt = "//"

    tester = Tester(controller=args.controller)

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
            if any([s in t for s in args.skip]):
                continue
            if args.only and not any([s in t for s in args.only]):
                continue
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
