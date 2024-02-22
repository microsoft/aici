# Type stubs

from __future__ import annotations
from typing import Any, Sequence, List
import pyaici.server as aici


def register(cb: aici.AiciCallbacks):
    """
    Use aici.start() instead.
    """
    ...


def tokenize(text: bytes | str) -> List[int]:
    """
    Return token indices for a given string (or byte sequence).
    """
    ...


def detokenize(tokens: List[int]) -> bytes:
    """
    Return byte (~string) representation of a given list of token indices.
    """
    ...


def self_seq_id() -> int:
    """
    Return identifier of the current sequence.
    Most useful with fork_group parameter in mid_process() callback.
    Best use aici.fork() instead.
    """
    ...


def get_var(name: str) -> None | bytes:
    """
    Get the value of a shared variable.
    """
    ...


def set_var(name: str, value: bytes | str) -> None:
    """
    Set the value of a shared variable.
    """
    ...


def append_var(name: str, value: bytes | str) -> None:
    """
    Append to the value of a shared variable.
    """
    ...


def eos_token() -> int:
    """
    Index of the end of sequence token.
    """
    ...


class TokenSet(Sequence[bool]):
    """
    Represents a set of tokens.
    The value is true at indices corresponding to tokens in the set.
    """

    def __init__(self):
        """
        Create an empty set (with len() set to the total number of tokens).
        """
        ...

    def __getitem__(self, i: int) -> bool:
        ...

    def __setitem__(self, i: int, v: bool) -> bool:
        ...

    def __len__(self) -> int:
        """
        Number of all possible tokens (whether they are in the set or not).
        """
        ...

    def set_all(self, value: bool):
        """
        Include or exclude all tokens from the set.
        """
        ...

    def num_set(self) -> int:
        """
        Number of tokens in the set.
        """
        ...

class Constraint:
    def __init__(self):
        """
        Initialize a constraint that allows any token.
        """
        ...

    def eos_allowed(self) -> bool:
        """
        Check if the constraint allows the generation to end at the current point.
        """
        ...

    def eos_forced(self) -> bool:
        """
        Check if the constraint forces the generation to end at the current point.
        """
        ...

    def token_allowed(self, t: int) -> bool:
        """
        Check if token `t` is allowed by the constraint.
        """
        ...

    def append_token(self, t: int):
        """
        Update the internal state of the constraint to reflect that token `t` was appended.
        """
        ...

    def allow_tokens(self, ts: TokenSet):
        """
        Set ts[] to True at all tokens that are allowed by the constraint.
        """
        ...


class RegexConstraint(Constraint):
    """
    A constraint that allows only tokens that match the regex.
    The regex is implicitly anchored at the start and end of the generation.
    """

    def __init__(self, pattern: str):
        ...


class CfgConstraint(Constraint):
    """
    A constraint that allows only tokens that match the specified yacc-like grammar.
    """

    def __init__(self, yacc_grammar: str):
        ...


class SubStrConstraint(Constraint):
    """
    A constraint that allows only word-substrings of given string.
    """

    def __init__(self, template: str, stop_at: str):
        ...


def is_server_side():
    """
    Return True if the code is running on the server.
    """
    # on server it's implemented natively, just like everything else is here
    return False

#
# Note, that this file is not embedded in pyctrl - it's only type stubs for a native module
#

print(f"""
This module is provided as part of the pyaici package to help with auto-completion in IDEs
while editing Python files to be uploaded to AICI server.

It will not work with the standard Python interpreter.

To upload and run a Python file on the server, use the following command:

    aici run myfile.py

Try 'aici run --help' for more info.

The 'aici' command can be replaced by 'python -m pyaici.cli' if needed.
""")
import sys
sys.exit(1)
