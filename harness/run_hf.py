import argparse

from typing import cast, Optional, Union
import torch
import time

from transformers import (
    GPT2Tokenizer,
    AutoTokenizer,
    PreTrainedModel,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

from transformers.generation.streamers import BaseStreamer

from biascomms import BiasServerComms

device = "cuda" if torch.cuda.is_available() else "cpu"


class AsyncLogitProcessor(LogitsProcessor, BaseStreamer):
    def __init__(self, comms: BiasServerComms) -> None:
        super().__init__()
        self.comms = comms
        self._idx = 0

    def put(self, value: torch.LongTensor):
        if self._idx == 0:
            self.comms.append_prompt_tokens(value)
        else:
            self.comms.append_tokens(value)
        self._idx += 1

    def end(self):
        self._idx = 0
        self.comms.reset()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._idx == 50:
            self._start_time = time.perf_counter()
        if self._idx == 150:
            print("Time", time.perf_counter() - self._start_time)
        bias_tensor = self.comms.get_bias_tensor()
        assert scores.shape[0] == 1 and scores.shape[1] == bias_tensor.shape[0]
        assert input_ids.shape[0] == 1  # and self._idx == input_ids.shape[1]
        return cast(
            torch.FloatTensor, bias_tensor.reshape(scores.shape).to(device) + scores
        )


model_name = "gpt2"
quant = False
# model_name = "OpenAssistant/llama2-13b-orca-8k-3319"


def main(args):
    if not args.tokenizer:
        args.tokenizer = args.model
    tokenizer = cast(PreTrainedTokenizer, AutoTokenizer.from_pretrained(args.tokenizer))
    
    if quant:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    model = cast(PreTrainedModel, model)
    # wproc = WasmProcessor(["./built/genbias"], tokenizer)
    comms = BiasServerComms(["wasmtime", "built/genbias.wasm"], tokenizer.vocab_size)
    wproc = AsyncLogitProcessor(comms)
    inp = tokenizer("A list of niceeee colors: red, blue", return_tensors="pt")
    proc = LogitsProcessorList()
    proc.append(wproc)
    output = model.generate(
        input_ids=inp.input_ids.to(device),
        attention_mask=inp.attention_mask.to(device),
        logits_processor=proc,
        streamer=wproc,
        max_new_tokens=160,
    )
    r = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    print("\n\n" + r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using HF Transformers with gvmrt"
    )
    parser.add_argument("--model", type=str, required=True, help="model to use")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="tokenizer to use; defaults to model name",
    )
    parser.add_argument("--gvm-module", type=str, default="", help="module id")
    parser.add_argument("--gvm-rt", type=str, required=True, help="module id")
    parser.add_argument(
        "--gvm-tokenizer",
        type=str,
        default="llama",
        help="tokenizer to use; llama, gpt4, ...",
    )
    args = parser.parse_args()
    main(args)
