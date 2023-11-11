#!/usr/bin/env python
import pyaici
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    pyaici.add_cli_args(args, single=False)
    args = args.parse_args()
    runner = pyaici.AiciRunner.from_cli(args)
    runner.bench()
