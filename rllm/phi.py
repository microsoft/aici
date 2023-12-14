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

# modeln = "microsoft/phi-1_5"
modeln = "./tmp/phi"
prompt = '''```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """'''

out_tokens = []
output = {}


class AsyncLogitProcessor(LogitsProcessor, BaseStreamer):
    def __init__(self) -> None:
        super().__init__()
        self.scores = None
        self._idx = 0

    def put(self, value: torch.LongTensor):
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


def main():
    if platform.system() == "Darwin":
        torch.set_default_device("cpu")
    else:
        torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(modeln, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(modeln, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

    wproc = AsyncLogitProcessor()
    proc = LogitsProcessorList()
    proc.append(wproc)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        logits_processor=proc,
        streamer=wproc,
        do_sample=True,
    )
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
    output["output"] = torch.tensor(out_tokens, dtype=torch.long)
    save_file(output, "tmp/reference.safetensors")


main()
