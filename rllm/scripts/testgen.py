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

N_LOGITS = 128
N_TOKENS = 30

models = {
    "phi-1_5": "microsoft/phi-1_5",
    "codellama": "codellama/CodeLlama-13b-Instruct-hf",
}

modelid = "codellama"
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
        save_test_case(f"expected/{modelid}/{k}.safetensors")


def load_safe(fn):
    inp = {}
    with safe_open(fn, framework="pt", device="cpu") as f:
        for key in f.keys():
            inp[key] = f.get_tensor(key)
    return inp


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
    if n <= 0: return
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

parser_show = subparsers.add_parser("show", help="Inspect a file")
parser_show.add_argument("file", type=str, help="Path to the file")

parser_truncate = subparsers.add_parser("truncate", help="Truncate a file")
parser_truncate.add_argument("file", type=str, help="Path to the file")
parser_truncate.add_argument("length", type=int, help="Length to truncate to")
parser_truncate.add_argument(
    "-w", "--write", action="store_true", help="Actually write the file"
)

args = parser.parse_args()

subc = args.subcommand
if subc == "generate":
    gen_output()
elif subc == "truncate":
    trunc(args.length, args.file, args.write)
elif subc == "show":
    show(args.file)
else:
    raise ValueError()
