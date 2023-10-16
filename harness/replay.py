#!/usr/bin/env python
import argparse
import pyaici


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using HF Transformers with aicirt"
    )
    parser.add_argument(
        "trace", help="path to JSONL trace file (generated with --aici-trace)"
    )
    pyaici.add_cli_args(parser)
    args = parser.parse_args()
    aici = pyaici.AiciRunner.from_cli(args)
    aici.replay(args.trace)
