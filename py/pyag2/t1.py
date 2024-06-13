import pyag2
import guidance
import pyag2._lib

import guidance.models._tokenizer


def main():
    # m = guidance.models.Transformers(model="../../tmp/Phi-3-mini-128k-instruct/", trust_remote_code=True)
    m = guidance.models.LlamaCpp(model="../../tmp/Phi-3-mini-4k-instruct-q4.gguf")
    t: guidance.models._tokenizer.Tokenizer = m.engine.tokenizer 
    tok = pyag2._lib.Ag2Tokenizer(t.eos_token_id, t.tokens)
    toks = tok.greedy_tokenize("Hello, world!")
    print(toks, tok.dbg_tokens(toks))


if __name__ == "__main__":
    main()
