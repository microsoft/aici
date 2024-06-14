from typing import List, Tuple, Mapping, Optional, Sequence, Union
from ._util import TokenId
from ._tokenizer import TokenizerWrapper

class LLTokenizer:
    vocab_size: int
    eos_token: TokenId

    def __new__(
        cls,
        tokenizer: TokenizerWrapper,
    ) -> "LLTokenizer":
        """
        Create a new tokenizer.
        """

    def greedy_tokenize(self, text: str) -> List[int]:
        """
        Tokenize the text using a greedy algorithm.
        This will not necesserily match BPE.
        """

    def tokenize_bytes(self, utf8bytes: bytes) -> List[int]:
        """
        Tokenize the text as bytes.
        This will use the underlaying Python tokenizer to tokenize valid UTF8
        prefix of the text, and then fallback to greedy_tokenize() for the last
        few bytes.
        """
    
    def tokenize_str(self, text: str) -> List[int]:
        """
        Same as tokenize_bytes, but for strings.
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

class LLInterpreter:
    def __new__(
        cls,
        tokenizer: LLTokenizer,
        llguidance_json: str,
        log_level: int = 1,
    ) -> "LLInterpreter":
        """
        Create a new interpreter.
        Args:
            tokenizer: LLTokenizer - the tokenizer to use
            llguidance_json: str - the JSON representation of the AG2 grammar/constraint
            log_level: int - the verbosity level of the interpreter
                0 is silent, 1 is warnings, 2 is verbose
        """

    def deep_copy(self) -> "LLInterpreter":
        """
        Create a deep copy of the interpreter.
        """

    def process_prompt(self, prompt: List[TokenId]) -> List[TokenId]:
        """
        Perform any adjustments to the prompt before completion.
        Returns the adjusted prompt.
        """

    def mid_process(
        self, backtrack: int, tokens: List[TokenId]
    ) -> Tuple[Optional[bytes], str]:
        """
        Perform next parsing step.
        Returns: optional token mask and a JSON string.
        """
