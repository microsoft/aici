import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation.streamers import BaseStreamer
from safetensors.torch import save_file
import platform
from safetensors import safe_open
import sys
import argparse
import os
import re
import ujson
import subprocess

N_LOGITS = 128
N_TOKENS = 30

models = {
    "phi-1_5": "microsoft/phi-1_5",
    "phi-2": "microsoft/phi-2",
    "codellama": "codellama/CodeLlama-13b-Instruct-hf",
    "codellama34": "codellama/CodeLlama-34b-Instruct-hf",
    "llama": "NousResearch/Llama-2-7b-hf",
    "orca": "microsoft/Orca-2-13b",
}

modelid = "llama"
modeln = models[modelid]

prompts = {
    #
    "primes": '''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
    #
    "lighthouse": """Write a detailed analogy between mathematics and a lighthouse.

Answer:""",
    #
    "cats": "If all cats are mzx and Fabian is a cat, then Fabian is",
}


out_tokens = []
output = {}


class AsyncLogitProcessor(LogitsProcessor, BaseStreamer):
    def __init__(self) -> None:
        super().__init__()
        self.scores = None
        self._idx = 0

    def put(self, value: torch.LongTensor):
        print("gen", self._idx)
        if self._idx == 0:
            output["prompt"] = value[0].clone()
        else:
            out_tokens.append(value.tolist()[0])
            self.scores = None
        self._idx += 1

    def end(self):
        self._idx = 0

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # print(bias_tensor.shape, scores.shape, input_ids.shape)
        assert input_ids.shape[0] == 1  # and self._idx == input_ids.shape[1]
        output["logits_%d" % (self._idx - 1)] = scores[0].clone().to(torch.half)
        return scores


def load_tokenizer():
    if platform.system() == "Darwin":
        torch.set_default_device("cpu")
    else:
        torch.set_default_device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(modeln, trust_remote_code=True)
    return tokenizer


def gen_output():
    global output, out_tokens

    tokenizer = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(modeln, trust_remote_code=True)

    wproc = AsyncLogitProcessor()
    proc = LogitsProcessorList()
    proc.append(wproc)

    for k in prompts.keys():
        output = {}
        out_tokens = []
        inputs = tokenizer(prompts[k], return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(
            **inputs,
            max_new_tokens=N_TOKENS,
            logits_processor=proc,
            streamer=wproc,
            do_sample=True,
        )
        text = tokenizer.batch_decode(outputs)[0]
        print(text)
        output["output"] = torch.tensor(out_tokens, dtype=torch.long)
        save_file(output, "tmp/reference.safetensors")
        os.makedirs(f"expected/{modelid}", exist_ok=True)
        save_test_case(f"expected/{modelid}/{k}.safetensors")


def load_safe(fn):
    inp = {}
    with safe_open(fn, framework="pt", device="cpu") as f:
        for key in f.keys():
            inp[key] = f.get_tensor(key)
    return inp


# tokens: [ '':1, ' If':960, ' all':599, ' c':274, 'ats':1446, ' are':526, ' m':286, 'z':29920, 'x':29916, ' and':322, ' Fab':10629, 'ian':713, ' is':338, ' a':263, ' cat':6635, ',':29892, ' then':769, ' Fab':10629, 'ian':713, ' is':338 ]
#  If all cats are mzx and Fabian is a cat, then Fabian is
# top 10 candidates:
#  -   286: '           m'   14.661 0.731229
#  -   263: '           a'   13.354 0.197962
#  -   341: '           M'   11.304 0.025479
#  -   884: '        also'   11.264 0.024478
#  -   385: '          an'   11.104 0.020853
#  -   451: '         not'    7.468 0.009020
#  -   611: '          ma'    6.674 0.004076
#  - 29871: '            '    6.466 0.003148
#  -   278: '         the'    6.440 0.003067
#  -   503: '           z'    6.340 0.002776
# sampled token:   286: ' m'

d = {
    "token": 286,
    "candidates": [
        {"token": 286, "logit": 14.661, "prob": 0.731229},
        {"token": 263, "logit": 13.354, "prob": 0.197962},
        {"token": 341, "logit": 11.304, "prob": 0.025479},
        {"token": 884, "logit": 11.264, "prob": 0.024478},
        {"token": 385, "logit": 11.104, "prob": 0.020853},
        {"token": 451, "logit": 7.468, "prob": 0.009020},
        {"token": 611, "logit": 6.674, "prob": 0.004076},
        {"token": 29871, "logit": 6.466, "prob": 0.003148},
        {"token": 278, "logit": 6.440, "prob": 0.003067},
        {"token": 503, "logit": 6.340, "prob": 0.002776},
    ],
}


def llamagen():
    for k in prompts.keys():
        out = subprocess.check_output(
            [
                "./tmp/llama.cpp/build/bin/main",
                "-m",
                "tmp/llama.cpp/models/codellama-34b-instruct.Q6_K.gguf",
                "--top-k", "30000", # force sorting of probs
                "--prompt",
                prompts[k],
                "-n",
                str(N_TOKENS),
            ],
            text=True,
        )
        log = out.split("\n")
        if not [l for l in log if "TOK:" in l]:
            print(out)
            sys.exit(1)
        os.makedirs(f"expected/{modelid}", exist_ok=True)
        llama_cpp_log_to_test_case_inner(log, f"expected/{modelid}/{k}.safetensors")


def llama_cpp_log_to_test_case_inner(lines: list[str], fn="tmp/compr.safetensors"):
    prompt = None
    tokens = []
    logits = []
    output = []
    prob_mass = []
    for line in lines:
        # print("LINE", line)
        if line.startswith("PROMPT: "):
            xtokens = ujson.decode(line[8:])
            prompt = torch.tensor(xtokens, dtype=torch.int32)
        elif line.startswith("TOK: "):
            d = ujson.decode(line[5:])
            output.append(d["token"])
            tokens.append([e["token"] for e in d["candidates"]])
            logits.append([e["logit"] for e in d["candidates"]])
            prob_mass.append(sum([e["prob"] for e in d["candidates"]]))

    out = {
        "prompt": prompt,
        "output": torch.tensor(output, dtype=torch.int32),
        "prob_mass": torch.tensor(prob_mass, dtype=torch.float32),
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "logits": torch.tensor(logits, dtype=torch.half),
    }
    print("Save", fn)
    save_file(out, fn)


def llama_cpp_log_to_test_case(log_fn: str, fn="tmp/compr.safetensors"):
    with open(log_fn, "r") as f:
        llama_cpp_log_to_test_case_inner(f.readlines(), fn)


def save_test_case(fn="tmp/compr.safetensors"):
    inp = load_safe("tmp/reference.safetensors")
    tokens = []
    logits = []
    prob_mass = []
    for i in range(1000):
        l: torch.Tensor = inp.get("logits_%d" % i, None)
        if l is None:
            break
        print(".", end="")
        sys.stdout.flush()
        ll = l.tolist()
        with_id = list(zip(ll, range(len(ll))))
        with_id.sort()
        with_id.reverse()
        l = torch.tensor([p for (p, _) in with_id])
        toks = torch.tensor([idx for (_, idx) in with_id], dtype=torch.int32)
        prob = l.softmax(0)[0:N_LOGITS].sum().tolist()
        prob_mass.append(prob)
        logits.append(l[0:N_LOGITS])
        tokens.append(toks[0:N_LOGITS])
        # print(N, prob, toks[0:N], with_id[0:N])
        # break

    pm = torch.tensor(prob_mass, dtype=torch.float32)
    print("\n", N_LOGITS, pm.min(), pm.mean(), pm.max())

    out = {
        "prompt": inp["prompt"].to(torch.int32),
        "output": inp["output"].to(torch.int32),
        "prob_mass": torch.tensor(prob_mass, dtype=torch.float32),
        "tokens": torch.stack(tokens, 0).to(torch.int32),
        "logits": torch.stack(logits, 0).to(torch.half),
    }
    save_file(out, fn)


def tinfo(lbl: str, ll: torch.Tensor):
    print(
        lbl + ":",
        ll.shape,
        "min:",
        ll.min(),
        "avg:",
        ll.mean(),
        "max:",
        ll.max(),
    )


def show(fn: str, n=0):
    tokenizer = load_tokenizer()
    inp = load_safe(fn)
    print("Prompt:", repr(tokenizer.decode(inp["prompt"])))
    print("Output:", repr(tokenizer.decode(inp["output"])))
    if n > 0:
        n = min(inp["output"].numel(), n)
        print("Trunc Output:", repr(tokenizer.decode(inp["output"][0:n])))
    elif n < 0:
        for k in range(inp["output"].numel()):
            print("Trunc to %d:" % k, repr(tokenizer.decode(inp["output"][0:k])))
    tinfo("logits", inp["logits"].to(torch.float))
    tinfo("prob_mass", inp["prob_mass"])
    return inp, n


def trunc(n: int, fn: str, wr: str):
    inp, n = show(fn, n)
    if n <= 0:
        return
    if wr:
        print("Writing", fn)
        out = {
            "prompt": inp["prompt"],
            "output": inp["output"][0:n],
            "prob_mass": inp["prob_mass"][0:n],
            "tokens": inp["tokens"][0:n, 0:N_LOGITS].contiguous(),
            "logits": inp["logits"][0:n, 0:N_LOGITS].contiguous(),
        }
        save_file(out, fn)


parser = argparse.ArgumentParser(
    prog="testgen.py", description="Generate and manipulate LLM testcases"
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="codellama",
    help="Model to use: " + ", ".join(models.keys()),
)

subparsers = parser.add_subparsers(dest="subcommand", required=True)

parser_gen = subparsers.add_parser("generate", help="Generate responses")
parser_gen.add_argument("--llama", action="store_true", help="Use llama.cpp")

parser_show = subparsers.add_parser("show", help="Inspect a file")
parser_show.add_argument("file", type=str, help="Path to the file")

parser_truncate = subparsers.add_parser("truncate", help="Truncate a file")
parser_truncate.add_argument("file", type=str, help="Path to the file")
parser_truncate.add_argument("length", type=int, help="Length to truncate to")
parser_truncate.add_argument(
    "-w", "--write", action="store_true", help="Actually write the file"
)

parser_llamacpp = subparsers.add_parser(
    "llamacpp", help="Convert llama.cpp log to testcase"
)
parser_llamacpp.add_argument("file", type=str, help="Path to the file llama.cpp log")


args = parser.parse_args()

modelid = args.model
modeln = models[modelid]

subc = args.subcommand
if subc == "generate":
    if args.llama:
        llamagen()
    else:
        gen_output()
elif subc == "truncate":
    trunc(args.length, args.file, args.write)
elif subc == "show":
    show(args.file)
elif subc == "llamacpp":
    llama_cpp_log_to_test_case(args.file)
else:
    raise ValueError()
