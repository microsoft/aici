#
# This is outdated
#

import argparse

from typing import cast, Optional, Union, List
import torch
import pyaici
import pyaici.comms

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


class StopGeneration(Exception):
    pass


class AsyncLogitProcessor(LogitsProcessor, BaseStreamer):
    def __init__(self, seq_id: int, runner: pyaici.comms.AiciRunner) -> None:
        super().__init__()
        self.runner = runner
        self._idx = 0
        self.seq_id = seq_id
        runner.add_mid(seq_id)
        runner.exec_mid()

    def _check_stop(self):
        to_stop = self.runner.get_seqs_to_stop()
        if self.seq_id in to_stop:
            raise StopGeneration

    def put(self, value: torch.LongTensor):
        if self._idx == 0:
            self._idx += 1
            return  # prompt
        runner = self.runner
        seq_id = self.seq_id
        runner.tokens_generated(seq_id, value.tolist())
        runner.exec_post_pre()
        runner.print_logs()

        self._check_stop()
        suspend, num_forks, ff_tokens = self.runner.pre_status(self.seq_id)
        assert not suspend, "forking not implemented"
        assert num_forks <= 1, "forking not implemented"
        assert len(ff_tokens) == 0, "ff_tokens not implemented"

        self._idx += 1
        runner.add_mid(seq_id)
        runner.exec_mid()

    def end(self):
        self._idx = 0
        self.runner.seq_freed(self.seq_id)
        self.seq_id += 1
        self.runner.flush_logit_bias()
        self.runner.print_logs()

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.Tensor
    ) -> torch.Tensor:
        runner = self.runner
        bias_tensor = runner.recv_logit_bias()
        runner.print_logs()
        self._check_stop()
        ff_tokens, backtrack = runner.mid_status(self.seq_id)
        assert backtrack == 0, "backtrack not implemented"
        assert len(ff_tokens) == 0, "ff_tokens not implemented"
        bias_tensor = torch.from_numpy(bias_tensor).to(scores.device).to(scores.dtype)
        # print(bias_tensor.shape, scores.shape, input_ids.shape)
        vocab_size = bias_tensor.shape[1]
        # scores should be the size of vocabulary but some models (phi-2) make it slightly bigger
        assert scores.shape[1] <= vocab_size + 1000
        scores = scores[:, 0:vocab_size]
        assert scores.shape == bias_tensor.shape
        assert input_ids.shape[0] == 1  # and self._idx == input_ids.shape[1]
        return bias_tensor + scores


def main(args):
    tokenizer = cast(
        PreTrainedTokenizer, AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = cast(PreTrainedModel, model)
    empty_tokens = cast(
        List[int], tokenizer.convert_tokens_to_ids(tokenizer.tokenize(""))
    )

    runner = pyaici.runner_from_cli(args)

    arg = ""
    if args.controller_arg:
        with open(args.controller_arg) as f:
            arg = f.read()
    req_id = "r1"
    seq_id = 1  # there can be multiple sequences in a single request
    runner.instantiate(req_id, empty_tokens, args.controller, arg)
    runner.assign_seq_id(req_id, seq_id)
    runner.print_logs()
    # we execute first post_pre here, so we get the initial ff_tokens
    runner.exec_post_pre()
    runner.print_logs()
    suspend, num_forks, ff_tokens = runner.pre_status(seq_id)
    to_stop = runner.get_seqs_to_stop()
    if seq_id in to_stop:
        print("AICI decided to stop")
        exit(1)
    assert not suspend, "forking not implemented"
    assert num_forks <= 1, "forking not implemented"

    prompt = torch.tensor(
        empty_tokens + ff_tokens, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    attn_mask = torch.ones(prompt.shape, dtype=torch.long, device=model.device)

    wproc = AsyncLogitProcessor(seq_id, runner)
    proc = LogitsProcessorList()
    proc.append(wproc)
    try:
        model.generate(
            input_ids=prompt,
            attention_mask=attn_mask,
            logits_processor=proc,
            streamer=wproc,
            max_new_tokens=2000,
            temperature=0.01,
            do_sample=True,
        )
        runner.print_logs()  # just in case
    except StopGeneration:
        runner.print_logs()
        print("AICI stop")


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
    pyaici.add_cli_args(parser, single=True)
    args = parser.parse_args()
    main(args)
