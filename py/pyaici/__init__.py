import argparse


def runner_from_cli(args, dtype: str = 'f32'):
    from pyaici.comms import AiciRunner

    tokenizer = args.aici_tokenizer

    # when no explicit --aici-tokenizer, we look for:
    #   --tokenizer + --tokenizer-revision
    #   --model + --revision
    if not tokenizer:
        model_tokenizer = getattr(args, "tokenizer", None)
        if model_tokenizer:
            rev = getattr(args, "tokenizer_revision", None)
            if rev:
                model_tokenizer += f"@{rev}"
            tokenizer = model_tokenizer
        else:
            model = getattr(args, "model", None)
            if model:
                rev = getattr(args, "revision", None)
                if rev:
                    model += f"@{rev}"
                tokenizer = model

    if not tokenizer:
        raise ValueError("No AICIrt tokenizer specified")
    if not args.aici_rt:
        raise ValueError("No AICIrt path specified")

    aici = AiciRunner(
        rtpath=args.aici_rt,
        tokenizer=tokenizer,
        trace_file=args.aici_trace,
        rtargs=args.aici_rtarg,
        pref=args.aici_shm_prefix,
        dtype=dtype,
    )
    return aici


def add_cli_args(parser: argparse.ArgumentParser, single=False):
    parser.add_argument(
        "--aici-rt",
        type=str,
        required=True,
        help="path to aicirt",
    )
    parser.add_argument(
        "--aici-shm-prefix",
        type=str,
        default="/aici0-",
        help="prefix for shared memory communication channels",
    )
    parser.add_argument(
        "--aici-tokenizer",
        type=str,
        default="",
        help=
        "tokenizer to use; llama, phi, ...; can also use HF tokenizer name",
    )
    parser.add_argument(
        "--aici-trace",
        type=str,
        help="save trace of aicirt interaction to a JSONL file",
    )
    parser.add_argument(
        "--aici-rtarg",
        "-A",
        type=str,
        default=[],
        action="append",
        help="pass argument to aicirt process",
    )

    if single:
        parser.add_argument(
            "--controller",
            type=str,
            required=True,
            help="id of the module to run",
        )
        parser.add_argument(
            "--controller-arg",
            type=str,
            default="",
            help="arg passed to module (filename)",
        )

    return parser
