import subprocess
import json
import sys
import os
import argparse

from . import rest, jssrc
from . import add_cli_args, runner_from_cli


def cli_error(msg: str):
    print("Error: " + msg)
    sys.exit(1)


def build_rust(folder: str):
    bin_file = ""
    spl = folder.split("::")
    if len(spl) > 1:
        folder = spl[0]
        bin_file = spl[1]
    r = subprocess.run(
        [
            "cargo",
            "metadata",
            "--offline",
            "--no-deps",
            "--format-version=1",
        ],
        cwd=folder,
        stdout=-1,
        check=True,
    )
    info = json.loads(r.stdout)
    if len(info["workspace_default_members"]) != 1:
        cli_error("please run from project, not workspace, folder")
    pkg_id = info["workspace_default_members"][0]
    pkg = [pkg for pkg in info["packages"] if pkg["id"] == pkg_id][0]

    bins = [trg for trg in pkg["targets"] if trg["kind"] == ["bin"]]
    if len(bins) == 0:
        cli_error("no bin targets found")
    bins_str = ", ".join([folder + "::" + trg["name"] for trg in bins])
    if bin_file:
        if len([trg for trg in bins if trg["name"] == bin_file]) == 0:
            cli_error(f"{bin_file} not found; try one of {bins_str}")
    else:
        if len(bins) > 1:
            cli_error("more than one bin target found; use one of: " + bins_str)
        bin_file = bins[0]["name"]
    print(f'will build {bin_file} from {pkg["manifest_path"]}')

    triple = "wasm32-wasi"
    trg_path = (
        info["target_directory"] + "/" + triple + "/release/" + bin_file + ".wasm"
    )
    # remove file first, so we're sure it's rebuilt
    try:
        os.unlink(trg_path)
    except:
        pass
    r = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--target",
            triple,
        ],
        cwd=folder,
    )
    if r.returncode != 0:
        sys.exit(1)
    bb = open(trg_path, "rb").read()
    M = 1024 * 1024
    print(f"built: {trg_path}, {len(bb)/M:.3} MiB")
    return rest.upload_module(trg_path)


def ask_completion(cmd_args, *args, **kwargs):
    if cmd_args is not None:
        for k in ["max_tokens", "prompt", "ignore_eos"]:
            v = getattr(cmd_args, k)
            if v is not None:
                kwargs[k] = v
    res = rest.completion(*args, **kwargs)
    print("\n[Prompt] " + res["request"]["prompt"] + "\n")
    for text in res["text"]:
        print("[Response] " + text + "\n")
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/response.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"response saved to {path}")
    print("Usage:", res["usage"])
    print("Storage:", res["storage"])


def infer_args(cmd: argparse.ArgumentParser):
    cmd.add_argument("--prompt", "-p", default="", type=str, help="specify prompt")
    cmd.add_argument(
        "--max-tokens", "-t", type=int, help="maximum number of tokens to generate"
    )


def save_file(name: str, content: str, force: bool):
    if os.path.exists(name) and not force:
        print(f"file {name} exists; use --force to overwrite")
        return
    with open(name, "w") as f:
        f.write(content)
    print(f"saved {name}")


def main_rest(args):
    if args.subcommand == "tags":
        for tag in rest.list_tags():
            print(rest.pp_tag(tag))
        return

    if args.subcommand == "infer":
        if args.prompt == "":
            cli_error("--prompt empty")
        # for plain prompting, use log-level 1 by default
        if args.log_level is None:
            rest.log_level = 1
        ask_completion(
            args,
            aici_module=None,
            aici_arg=None,
            max_tokens=100,
        )
        return

    aici_module = ""

    for k in ["build", "upload", "ctrl", "tag", "ignore_eos"]:
        if k not in args:
            setattr(args, k, None)

    if args.build:
        assert not aici_module
        aici_module = build_rust(args.build)

    if args.upload:
        assert not aici_module
        aici_module = rest.upload_module(args.upload)

    if args.ctrl:
        assert not aici_module
        aici_module = args.ctrl

    if args.tag:
        if len(aici_module) != 64:
            cli_error("no AICI Controller to tag")
        rest.tag_module(aici_module, args.tag)

    if args.subcommand == "run":
        aici_arg = ""
        fn: str = args.aici_arg
        if fn == "-":
            aici_arg = sys.stdin.read()
        elif fn is not None:
            aici_arg = open(fn).read()
            if not aici_module:
                if fn.endswith(".py"):
                    aici_module = "pyctrl-latest"
                elif fn.endswith(".js"):
                    aici_module = "jsctrl-latest"
                elif fn.endswith(".json"):
                    aici_module = "declctrl-latest"
                else:
                    cli_error(
                        "Can't determine AICI Controller type from file name: " + fn
                    )
                print(f"Running with tagged AICI Controller: {aici_module}")
        if not aici_module:
            cli_error("no AICI Controller specified to run")

        ask_completion(
            args,
            aici_module=aici_module,
            aici_arg=aici_arg,
            ignore_eos=True,
            max_tokens=2000,
        )


def main_inner():
    parser = argparse.ArgumentParser(
        description="Upload an AICI Controller and completion request to rllm or vllm",
        prog="aici",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=int,
        help="log level (higher is more); default 3 (except in 'infer', where it's 1)",
    )

    parser.add_argument(
        "--all-prefixes",
        action="store_true",
        help="attempt the action for all detected prefixes (models/deployments)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_cmd = subparsers.add_parser(
        "run",
        help="run model inference via an AICI Controller",
        description="Run model inference via an AICI Controller.",
        epilog="""
        If FILE ends with .py, --ctrl defaults to 'pyctrl-latest'.
        Similarly, it's 'jsctrl-latest' for .js and 'declctrl-latest' for .json.
        """,
    )
    run_cmd.add_argument(
        "aici_arg",
        metavar='FILE',
        nargs="?",
        help="file to pass to the AICI Controller; use '-' for stdin",
    )
    infer_args(run_cmd)
    run_cmd.add_argument(
        "--ctrl",
        "-c",
        metavar="MODULE_ID",
        type=str,
        help="tag name or hex module id (sha256 of .wasm file)",
    )
    run_cmd.add_argument(
        "--upload",
        "-u",
        metavar="WASM_FILE",
        type=str,
        help="path to .wasm file to upload; shorthand for 'aici upload WASM_FILE'",
    )
    run_cmd.add_argument(
        "--build",
        "-b",
        metavar="FOLDER",
        type=str,
        help="path to rust project to build and upload; shorthand for 'aici build FOLDER'",
    )

    infer_cmd = subparsers.add_parser(
        "infer",
        help="run model inference without any AICI Controller",
        description="Run model inference without any AICI Controller.",
    )
    infer_args(infer_cmd)
    infer_cmd.add_argument(
        "--ignore-eos", action="store_true", help="ignore EOS tokens generated by model"
    )

    tags_cmd = subparsers.add_parser(
        "tags",
        help="list module tags available on the server",
        description="List module tags available on the server.",
    )

    # don't give help= -> it's quite internal
    bench_cmd = subparsers.add_parser(
        "benchrt",
        description="benchmark the aicirt communication mechanisms",
    )
    add_cli_args(bench_cmd, single=False)

    upload_cmd = subparsers.add_parser(
        "upload",
        help="upload a AICI Controller to the server",
        description="Upload a AICI Controller to the server.",
    )
    upload_cmd.add_argument(
        "upload", metavar="WASM_FILE", help="path to .wasm file to upload"
    )

    build_cmd = subparsers.add_parser(
        "build",
        help="build and upload a AICI Controller to the server",
        description="Build and upload a AICI Controller to the server.",
    )
    build_cmd.add_argument(
        "build", metavar="FOLDER", help="path to rust project (folder with Cargo.toml)"
    )

    jsinit_cmd = subparsers.add_parser(
        "jsinit",
        help="intialize current folder for jsctrl",
        description="Intialize a JavaScript/TypeScript folder for jsctrl.",
    )
    jsinit_cmd.add_argument(
        "--force", "-f", action="store_true", help="overwrite existing files"
    )

    for cmd in [upload_cmd, build_cmd]:
        cmd.add_argument(
            "--tag",
            "-T",
            type=str,
            default=[],
            action="append",
            help="tag the AICI Controller after uploading; can be used multiple times to set multiple tags",
        )

    args = parser.parse_args()

    if args.log_level:
        rest.log_level = args.log_level
    else:
        rest.log_level = 3

    if args.subcommand == "jsinit":
        save_file("tsconfig.json", jssrc.tsconfig_json, args.force)
        save_file("aici-types.d.ts", jssrc.aici_types_d_t, args.force)
        save_file("hello.js", jssrc.hello_js, args.force)
        return

    if args.subcommand == "benchrt":
        runner_from_cli(args).bench()
        return

    if args.all_prefixes:
        prefixes = rest.detect_prefixes()
        if len(prefixes) <= 1:
            print("no prefixes detected; continuing")
            main_rest(args)
        else:
            base_url = rest.base_url
            pref0s = [p for p in prefixes if p in base_url]
            if len(pref0s) != 1:
                cli_error("can't determine prefix from base url")
            for prefix in prefixes:
                print(f"prefix: {prefix}")
                rest.base_url = base_url.replace(pref0s[0], prefix)
                main_rest(args)
    else:
        main_rest(args)


def main():
    try:
        main_inner()
    except RuntimeError as e:
        cli_error(str(e))


if __name__ == "__main__":
    main()
