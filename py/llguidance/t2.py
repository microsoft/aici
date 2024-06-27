from typing import Any
import tokenizers
import llguidance


class MockTokenizer:

    def _tokenize_str(self, s: str) -> list[int]:
        return self.hf_tokenizer.encode(s).ids

    def __init__(self) -> None:
        self.hf_tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )
        empty = self._tokenize_str("")
        if empty:
            self.bos_token_id = empty[0]
        else:
            self.bos_token_id = None
        eos = self._tokenize_str("</s>")
        assert len(eos) == 1
        self.eos_token_id = eos[0]
        self.tokens = []
        for i in range(self.hf_tokenizer.get_vocab_size()):
            s = self.hf_tokenizer.decode([i])
            if "\uFFFD" in s:
                t: str = self.hf_tokenizer.id_to_token(i)
                if t.startswith("<0x"):
                    self.tokens.append(bytes([int(t[3:5], 16)]))
                    continue
            self.tokens.append(s.encode("utf-8"))
        assert len(self.tokens) == self.hf_tokenizer.get_vocab_size()

    def __call__(self, s):
        return self._tokenize_str(s)


def main():
    t = llguidance.LLTokenizer(llguidance.TokenizerWrapper(MockTokenizer()))
    print(t.tokenize_str("Hello, world!"))
    print(t.tokenize_bytes(b"\xcf"))


main()
