import llguidance
import guidance
import json

import numpy as np

from guidance.models._tokenizer import Tokenizer
from guidance.models.llama_cpp._llama_cpp import LlamaCppEngine

from typing import List


def softmax(logits: np.ndarray, temperature=1.0) -> np.ndarray:
    # Adjust logits by temperature
    adjusted_logits = logits / temperature
    # Compute softmax
    exp_logits = np.exp(adjusted_logits - np.max(adjusted_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

def sample_with_temperature(logits: np.ndarray, temperature=1.0):
    if temperature < 0.0001:
        return int(np.argmax(logits))
    # Get probabilities from softmax
    probabilities = softmax(logits, temperature)
    # Sample an index based on the probabilities
    sampled_index = np.random.choice(len(logits), p=probabilities)
    return sampled_index

def run_constraint(tok: llguidance.Ag2Tokenizer, e: LlamaCppEngine, grm: guidance.GrammarFunction):
    max_tokens = 100
    serialized = grm.llguidance_serialize()
    serialized["max_tokens"] = max_tokens
    interp = llguidance.Ag2Interpreter(tok, json.dumps(serialized))
    tokens = []
    if e.tokenizer.bos_token_id is not None:
        tokens.append(e.tokenizer.bos_token_id)
    tokens = interp.process_prompt(tokens)
    backtrack = 0
    step_tokens = []
    for _ in range(max_tokens):
        mask, resp = interp.mid_process(backtrack, step_tokens)
        r = json.loads(resp)
        progress: List[dict] = r["progress"]
        for p in progress:
            print(p)
        if r["stop"]:
            break
        backtrack: int = r["backtrack"]
        step_tokens: List[int] = r["ff_tokens"]
        if mask is not None:
            assert backtrack == 0
            assert len(step_tokens) == 0
            logits = e.get_logits(tokens, None, None)
            logits += np.frombuffer(mask, dtype=np.uint8)
            tok_idx = sample_with_temperature(logits, r["temperature"])
            tokens.append(tok_idx)
            step_tokens = [tok_idx]
        else:
            if backtrack:
                del tokens[-backtrack:]
            tokens += step_tokens




def main():
    # m = guidance.models.Transformers(model="../../tmp/Phi-3-mini-128k-instruct/", trust_remote_code=True)
    m = guidance.models.LlamaCpp(model="../../tmp/Phi-3-mini-4k-instruct-q4.gguf")
    t: Tokenizer = m.engine.tokenizer
    tok = llguidance.Ag2Tokenizer(t.eos_token_id, t.tokens)
    run_constraint(tok, m.engine, "Here's a joke: " + guidance.gen(regex="[a-z ]+", stop="\n"))


if __name__ == "__main__":
    main()
