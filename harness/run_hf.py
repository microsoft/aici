import argparse

from typing import cast, Optional, Union
import torch
import time
import pyaici

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)

from transformers.generation.streamers import BaseStreamer

device = "cuda" if torch.cuda.is_available() else "cpu"


class AsyncLogitProcessor(LogitsProcessor, BaseStreamer):
    def __init__(
        self, runner: pyaici.AiciRunner, module_id: str, module_arg: str
    ) -> None:
        super().__init__()
        self.runner = runner
        self._idx = 0
        self.wasm_id = 1
        self.module_id = module_id
        self.module_arg = module_arg

    def put(self, value: torch.LongTensor):
        if self._idx == 0:
            self.runner.step_add_prompt(
                self.wasm_id, value[0].tolist(), self.module_id, self.module_arg
            )
        else:
            self.runner.step_add_token(self.wasm_id, value[0].tolist())
        self.runner.step_finish()
        self._idx += 1

    def end(self):
        self._idx = 0
        self.wasm_id += 1
        self.runner.flush_logit_bias()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        bias_tensor = self.runner.recv_logit_bias()
        bias_tensor = torch.from_numpy(bias_tensor).to(scores.device).to(scores.dtype)
        # print(bias_tensor.shape, scores.shape, input_ids.shape)
        assert scores.shape == bias_tensor.shape
        assert input_ids.shape[0] == 1  # and self._idx == input_ids.shape[1]
        return bias_tensor + scores


def main(args):
    if not args.tokenizer:
        args.tokenizer = args.model
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(args.tokenizer))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = cast(PreTrainedModel, model)

    runner = pyaici.AiciRunner(rtpath=args.aici_rt, tokenizer=args.aici_tokenizer)

    wproc = AsyncLogitProcessor(runner, args.aici_module, args.aici_module_arg)
    inp = tokenizer(
        "Here is an example JSON about Joe Random Hacker in Seattle:\n",
        return_tensors="pt",
    )
    proc = LogitsProcessorList()
    proc.append(wproc)
    output = model.generate(
        input_ids=inp.input_ids.to(model.device),
        attention_mask=inp.attention_mask.to(model.device),
        logits_processor=proc,
        streamer=wproc,
        max_new_tokens=200,
        do_sample=True,
    )
    r = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print("\n\n" + r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using HF Transformers with aicirt"
    )
    parser.add_argument("--model", type=str, required=True, help="model to use")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="tokenizer to use; defaults to model name",
    )
    parser.add_argument(
        "--aici-module-arg", type=str, default="", help="arg passed to module"
    )
    parser.add_argument(
        "--aici-module", type=str, required=True, help="id of the module to run"
    )
    parser.add_argument("--aici-rt", type=str, required=True, help="path to aicirt")
    parser.add_argument(
        "--aici-tokenizer",
        type=str,
        default="llama",
        help="tokenizer to use; llama, gpt4, ...",
    )
    args = parser.parse_args()
    main(args)
