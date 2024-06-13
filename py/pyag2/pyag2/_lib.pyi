from typing import List, Tuple, Mapping, Optional, Sequence, Union
from ._util import TokenId

class Ag2Tokenizer:
    vocab_size: int
    eos_token: TokenId

    def __new__(
        cls,
        eos_token: TokenId,
        tokens: Sequence[bytes],
    ) -> "Ag2Tokenizer":
        """
        Create a new tokenizer.
        Args:
            eos_token: TokenId - the identifier of the end-of-sequence/end-of-text token
            tokens: Sequence[bytes] - maps regular tokens to their byte-string representation
                special tokens need to map to empty bytes
        """

    def greedy_tokenize(self, text: str) -> List[int]:
        """
        Tokenize the text using a greedy algorithm.
        This will not necesserily match BPE.
        """

    def dbg_tokens(self, tokens: List[int]) -> str:
        """
        Return a debug string representation of the tokens.
        The result is double-quoted and tokens are separated by '‧'.
        """

    def decode_str(self, tokens: List[int]) -> str:
        """
        Decode the tokens into a string.
        Invalid UTF-8 will be replaced with the Unicode replacement character.
        """

    def decode_bytes(self, tokens: List[int]) -> bytes:
        """
        Decode the tokens into a bytes object.
        """

class Ag2Interpreter:
    def __new__(
        cls,
        tokenizer: Ag2Tokenizer,
        ag2_json: str
    ) -> "Ag2Interpreter":
        """
        Create a new interpreter.
        Args:
            tokenizer: Ag2Tokenizer - the tokenizer to use
            ag2_json: str - the JSON representation of the AG2 grammar/constraint
        """

    def process_prompt(self, prompt: List[TokenId]) -> List[TokenId]:
        """
        Perform any adjustments to the prompt before completion.
        Returns the adjusted prompt.
        """

    def mid_process(self, backtrack: int, tokens: List[TokenId]) -> Tuple[Optional[bytes], str]:
        """
        Perform next parsing step.
        Returns: optional token mask and a JSON string.
        """
