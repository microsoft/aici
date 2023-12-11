#!/usr/bin/env python
import pyaici.rest
import tokenizers
import sys
import threading

def run_n(fn, n):
    threads = []
    for _ in range(n):
        t = threading.Thread(target=fn)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def main():
    prompt = open(sys.argv[1]).read()
    t = tokenizers.Tokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokens = t.encode(prompt).ids
    mode = "concat"

    if mode == "grow":
        l = 10
        while l < len(tokens):
            l = int(l * 1.1)
            prompt = t.decode(tokens[0:l])
            r = pyaici.rest.completion(prompt, max_tokens=1)
            print(l)
    elif mode == "concat":
        prompt1 = t.decode(tokens[0:800])
        prompt2 = t.decode(tokens[0:1600])
        def pp():
            pyaici.rest.completion(prompt1, max_tokens=1)
        if True:
            run_n(pp, 2)
        else:
            pyaici.rest.completion(prompt2, max_tokens=1)



if __name__ == "__main__":
    main()
