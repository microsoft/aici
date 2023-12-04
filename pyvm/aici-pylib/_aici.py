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
    ...


def detokenize(tokens: list[int]) -> bytes:
    ...


def get_var(name: str) -> None | bytes:
    ...


def set_var(name: str, value: bytes | str) -> None:
    ...


def append_var(name: str, value: bytes | str) -> None:
    ...

def eos_token() -> int:
    ...

class TokenSet(Sequence[bool]):
    def __init__(self):
        ...

    def __getitem__(self, i: int) -> bool:
        ...

    def __len__(self) -> int:
        ...

    def set_all(self, value: bool):
        ...


class Constraint:
    def eos_allowed(self) -> bool:
        ...

    def token_allowed(self, t: int) -> bool:
        ...

    def append_token(self, t: int):
        ...

    def allow_tokens(self, ts: TokenSet):
        ...
    
    @classmethod
    def regex(cls, pattern: str) -> Constraint:
        ...
