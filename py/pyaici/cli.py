import subprocess
import json
import sys
import os
import argparse

from . import rest, jssrc
from . import add_cli_args, runner_from_cli

from typing import List, Dict, Any, Optional


def cli_error(msg: str):
    print("Error: " + msg)
    sys.exit(1)


def build_rust(folder: str, features: List[str] = []):
    bin_file = ""
    spl = folder.split("::")
    if len(spl) > 1:
        folder = spl[0]
        bin_file = spl[1]
    if not os.path.exists(folder + "/Cargo.toml"):
        cli_error(f"{folder}/Cargo.toml not found")
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
            cli_error("more than one bin target found; use one of: " +
                      bins_str)
        bin_file = bins[0]["name"]
    print(f'will build {bin_file} from {pkg["manifest_path"]}')

    triple = "wasm32-wasip2"
    trg_path = (info["target_directory"] + "/" + triple + "/release/" +
                bin_file + ".wasm")
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
        ] + (["--features", ",".join(features)] if features else []),
        cwd=folder,
    )
    if r.returncode != 0:
        sys.exit(1)
    r = subprocess.run(
        [
            "wasm-tools",
            "component",
            "new",
            trg_path,
            "-o",
            component_path,
            "--adapt",
            reactor_path,
        ]
    )
    if r.returncode != 0:
        sys.exit(1)
    bb = open(trg_path, "rb").read()

    M = 1024 * 1024
    print(f"built: {component_path}, {len(bb)/M:.3} MiB")
    return rest.upload_module(component_path)


def run_ctrl(
    cmd_args,
    *,
    controller: str,
    controller_arg="",
    prompt="",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    print_response: bool = True,
):

    def attr(name: str, default: Any):
        r = getattr(cmd_args, name)
        if r is not None:
            return r
        return default

    max_tokens = attr("max_tokens", max_tokens)
    temperature = attr("temperature", temperature)
    res = rest.run_controller(
        controller=controller,
        controller_arg=controller_arg,
        temperature=temperature,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    if print_response:
        for text in res["text"]:
            print("[Response] " + text + "\n")
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/response.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"response saved to {path}")
    print("Usage:", res["usage"])
    print("Timing:", res["timing"])
    print("Tokens/sec:", res["tps"])
    print("Storage:", res["storage"])


def infer_args(cmd: argparse.ArgumentParser):
    cmd.add_argument("--max-tokens",
                     "-t",
                     type=int,
                     help="maximum number of tokens to generate")
    cmd.add_argument(
        "--temperature",
        type=float,
        help="temperature for sampling; default 0.0 (argmax)",
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
            cli_error("empty prompt")
        # for plain prompting, use log-level 1 by default
        if args.log_level is None:
            rest.log_level = 1
        run_ctrl(
            args,
            controller="none",
            controller_arg=args.prompt,
            max_tokens=100,
        )
        return

    controller = ""

    for k in ["build", "upload", "ctrl", "tag"]:
        if k not in args:
            setattr(args, k, None)

    if args.build:
        assert not controller
        controller = build_rust(args.build)

    if args.upload:
        assert not controller
        controller = rest.upload_module(args.upload)

    if args.ctrl:
        assert not controller
        controller = args.ctrl

    if args.tag:
        if len(controller) != 64:
            cli_error("no AICI Controller to tag")
        rest.tag_module(controller, args.tag)

    if args.subcommand == "run":
        controller_arg = ""
        fn: str = args.controller_arg
        if fn == "-":
            controller_arg = sys.stdin.read()
        elif fn is not None:
            controller_arg = open(fn).read()
            if not controller:
                if fn.endswith(".py"):
                    controller = "gh:microsoft/aici/pyctrl"
                elif fn.endswith(".js"):
                    controller = "gh:microsoft/aici/jsctrl"
                elif fn.endswith(".json"):
                    controller = "gh:microsoft/aici/declctrl"
                else:
                    cli_error(
                        "Can't determine AICI Controller type from file name: "
                        + fn)
                print(f"Running with tagged AICI Controller: {controller}")
        if not controller:
            cli_error("no AICI Controller specified to run")

        run_ctrl(
            args,
            controller=controller,
            controller_arg=controller_arg,
            prompt=args.prompt,
        )


def main_inner():
    parser = argparse.ArgumentParser(
        description=
        "Upload an AICI Controller and completion request to rllm or vllm",
        prog="aici",
    )

    parser.add_argument(
        "--log-level",
        "-l",
        type=int,
        help=
        "log level (higher is more); default 3 (except in 'infer', where it's 1)",
    )

    parser.add_argument(
        "--all-prefixes",
        action="store_true",
        help=
        "attempt the action for all detected prefixes (models/deployments)",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_cmd = subparsers.add_parser(
        "run",
        help="run model inference via an AICI Controller",
        description="Run model inference via an AICI Controller.",
        epilog="""
        Using --ctrl gh:user/repo/ctrl instructs the server to find *.wasm file containing string 'ctrl' in the latest release
        of user/repo on GitHub. The /ctrl part is optional if there is only one controller in the release.
        You can specify release with --ctrl gh:user/repo/ctrl/v1.2.3.

        If FILE ends with .py, --ctrl defaults to 'gh:microsoft/aici/pyctrl'.
        Similarly, it's 'gh:microsoft/aici/jsctrl' for .js and 'gh:microsoft/aici/declctrl' for .json.
        """,
    )
    run_cmd.add_argument("--prompt",
                         "-p",
                         type=str,
                         default="",
                         help="initial prompt if any")

    run_cmd.add_argument(
        "controller_arg",
        metavar="FILE",
        nargs="?",
        help="file to pass to the AICI Controller; use '-' for stdin",
    )
    infer_args(run_cmd)
    run_cmd.add_argument(
        "--ctrl",
        "-c",
        metavar="MODULE_ID",
        type=str,
        help="tag name, module id (sha256 of .wasm file), or gh:user/repo",
    )
    run_cmd.add_argument(
        "--upload",
        "-u",
        metavar="WASM_FILE",
        type=str,
        help=
        "path to .wasm file to upload; shorthand for 'aici upload WASM_FILE'",
    )
    run_cmd.add_argument(
        "--build",
        "-b",
        metavar="FOLDER",
        type=str,
        help=
        "path to rust project to build and upload; shorthand for 'aici build FOLDER'",
    )

    infer_cmd = subparsers.add_parser(
        "infer",
        help="run model inference without any AICI Controller",
        description="Run model inference without any AICI Controller.",
    )
    infer_cmd.add_argument("prompt", help="prompt to pass to the model")
    infer_args(infer_cmd)

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
    upload_cmd.add_argument("upload",
                            metavar="WASM_FILE",
                            help="path to .wasm file to upload")

    build_cmd = subparsers.add_parser(
        "build",
        help="build and upload a AICI Controller to the server",
        description="Build and upload a AICI Controller to the server.",
    )
    build_cmd.add_argument(
        "build",
        metavar="FOLDER",
        help="path to rust project (folder with Cargo.toml)")

    jsinit_cmd = subparsers.add_parser(
        "jsinit",
        help="initialize current folder for jsctrl",
        description="Initialize a JavaScript/TypeScript folder for jsctrl.",
    )
    jsinit_cmd.add_argument("--force",
                            "-f",
                            action="store_true",
                            help="overwrite existing files")

    for cmd in [upload_cmd, build_cmd]:
        cmd.add_argument(
            "--tag",
            "-T",
            type=str,
            default=[],
            action="append",
            help=
            "tag the AICI Controller after uploading; can be used multiple times to set multiple tags",
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

    if os.environ.get(rest.BASE_URL_ENV, None) is None:
        cli_error(f"environment variable {rest.BASE_URL_ENV} not set")

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
