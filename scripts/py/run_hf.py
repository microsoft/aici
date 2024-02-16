# This example embeds AICI in HuggingFace transformers using
# model.forward() and manual sampling.
# It supports backtracking and fast-forward tokens,
# but does not support forking.
#
# See ../scripts/hf.sh for how to run this.

import argparse

from typing import cast, Optional, Union, List
import torch
import pyaici
import pyaici.comms
from pyaici.comms import AiciRunner
from torch import nn

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class StopGeneration(Exception):
    pass


def check_stop(runner: AiciRunner, seq_id: int):
    to_stop = runner.get_seqs_to_stop()
    if seq_id in to_stop:
        raise StopGeneration


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
    req_id = "r1"  # arbitrary string
    seq_id = 1  # there can be multiple sequences in a single request
    res = runner.instantiate(req_id, empty_tokens, args.controller, arg)
    if isinstance(res, list):
        ff_tokens = res
    else:
        runner.print_logs_for(seq_id, res["forks"][0])
        exit(1)

    runner.assign_seq_id(req_id, seq_id)
    runner.print_logs()

    to_stop = runner.get_seqs_to_stop()
    if seq_id in to_stop:
        print("AICI decided to stop")
        exit(1)

    prompt = torch.tensor(
        empty_tokens + ff_tokens, dtype=torch.long, device=model.device
    ).unsqueeze(0)

    model_kwargs = {
        "attention_mask": None,
        "use_cache": True,
    }
    input_ids = prompt.squeeze(0)
    temperature = 0.01

    try:
        for _ in range(2000):
            runner.add_mid(seq_id)
            runner.exec_mid()

            m_inp = input_ids.unsqueeze(0)
            model_kwargs["attention_mask"] = torch.ones(
                m_inp.shape, dtype=torch.long, device=model.device
            )
            model_inputs = model.prepare_inputs_for_generation(m_inp, **model_kwargs)
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            scores: torch.Tensor = outputs.logits[:, -1, :]

            bias_tensor = runner.recv_logit_bias()
            runner.print_logs()
            check_stop(runner, seq_id)
            ff_tokens, backtrack = runner.mid_status(seq_id)
            assert backtrack == 0, "backtrack not implemented"
            assert len(ff_tokens) == 0, "ff_tokens not implemented"
            bias_tensor = (
                torch.from_numpy(bias_tensor).to(scores.device).to(scores.dtype)
            )
            # print(bias_tensor.shape, scores.shape, input_ids.shape)
            vocab_size = bias_tensor.shape[1]
            # scores should be the size of vocabulary but some models (phi-2) make it slightly bigger
            assert scores.shape[1] <= vocab_size + 1000
            scores = scores[:, 0:vocab_size]
            assert scores.shape == bias_tensor.shape
            scores += bias_tensor
            scores /= temperature

            probs = nn.functional.softmax(scores, dim=-1)

            if backtrack > 0 or len(ff_tokens) > 0:
                next_tokens = torch.tensor(
                    ff_tokens, dtype=torch.long, device=model.device
                )
            else:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            runner.tokens_generated(seq_id, next_tokens.tolist())
            runner.exec_post_pre()
            runner.print_logs()
            check_stop(runner, seq_id)

            if backtrack > 0:
                input_ids = input_ids[:-backtrack]
            computed_kv_len = input_ids.shape[0]
            input_ids = torch.cat([input_ids, next_tokens], dim=0)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if "past_key_values" in model_kwargs:
                m = model_kwargs["past_key_values"]
                # len(m) == num_layers, len(m[0]) == 2 (key, value)
                # shape of each elt is (batch_size, num_heads, seq_len, head_dim)
                if m is not None and backtrack > 0:
                    m = [
                        (
                            q[0][:, :, 0:computed_kv_len, :],
                            q[1][:, :, 0:computed_kv_len, :],
                        )
                        for q in m
                    ]
                    model_kwargs["past_key_values"] = m

            suspend, num_forks, ff_tokens = runner.pre_status(seq_id)
            check_stop(runner, seq_id)
            assert not suspend, "forking not implemented"
            assert num_forks <= 1, "forking not implemented"

            if len(ff_tokens) > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor(ff_tokens, dtype=torch.long, device=model.device),
                    ],
                    dim=0,
                )

    except StopGeneration:
        runner.print_logs()
        print("AICI decided to stop")


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
