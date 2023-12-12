# type stubs
from __future__ import annotations
from typing import Any, Sequence
import aici


def register(cb: aici.AiciCallbacks):
    """
    Use aici.start() instead.
    """
    ...


def tokenize(text: bytes | str) -> list[int]:
    """
    Return token indices for a given string (or byte sequence).
    """
    ...


def detokenize(tokens: list[int]) -> bytes:
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
    The value is true at indicies corresponding to tokens in the set.
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
        Number of all tokens (not only in the set).
        """
        ...

    def set_all(self, value: bool):
        """
        Include or exclude all tokens from the set.
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
