import torch

from typing import List, Dict, Optional

from vllm.sequence import SamplingController, SamplerOutput
from vllm.sampling_params import SamplingParams
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .comms import AiciRunner
from ._vllm_protocol import EPSILON_TEMP


class AiciSamplingController(SamplingController):

    def __init__(self, aici_runner: AiciRunner):
        super().__init__()
        self.runner = aici_runner
        self.logit_pending = False
        self.seq_id_to_sampling_idx: Dict[int, int] = {}
        self.req_id_to_seq_id: Dict[str, int] = {}
        self.seq_id_to_sampling_params: Dict[int, SamplingParams] = {}

    def resolve_req_id(self, req_id: str) -> Optional[int]:
        return self.req_id_to_seq_id.get(req_id)


    def log(self, msg: str):
        """Log message to stdout."""
        print(f"AICI: {msg}")

    def empty_step(self):
        self.log("empty_step")
        runner = self.runner
        runner.add_mid_for_finished()
        if runner.needs_exec_mid():
            self.log("empty_step EXEC")
            runner.exec_mid()
            _ = self.runner.recv_logit_bias_torch()
            self.log("empty_step STOP")

    def prepare(self, sampling_metadata: "SamplingMetadata"):
        self.log("prepare")
        runner = self.runner
        seq_id_to_sampling_idx: Dict[int, int] = {}
        seq_id_to_sampling_params: Dict[int, SamplingParams] = {}
        req_id_to_seq_id: Dict[str, int] = {}
        for group in sampling_metadata.seq_groups:
            if not getattr(group.sampling_params, "has_aici", False):
                continue
            assert len(group.seq_ids) == 1
            seq_id = group.seq_ids[0]
            seq_id_to_sampling_params[seq_id] = group.sampling_params
            req_id_to_seq_id[group.request_id] = seq_id
            sample_indices = group.sample_indices
            if sample_indices:
                assert len(sample_indices) == 1
                runner.assign_seq_id(group.request_id, seq_id, optional=True)
                runner.add_mid(seq_id)
                seq_id_to_sampling_idx[seq_id] = sample_indices[0]
            else:
                pass
        runner.add_mid_for_finished()
        if runner.needs_exec_mid():
            runner.exec_mid()
            self.seq_id_to_sampling_idx = seq_id_to_sampling_idx
            self.seq_id_to_sampling_params = seq_id_to_sampling_params
            self.req_id_to_seq_id = req_id_to_seq_id
            self.logit_pending = True

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        _num_tokens, vocab_size = logits.shape
        if not self.logit_pending:
            return logits
        self.logit_pending = False
        resp, bias = self.runner.recv_logit_bias_torch()
        bias = bias.to(logits.device)  # TODO use non_blocking?
        sampling_map = self.seq_id_to_sampling_idx
        _num_masks, vocab_size2 = bias.shape
        assert vocab_size2 >= vocab_size, f"{vocab_size2} != {vocab_size}"
        for seq_id, mid_res in resp.items():
            if mid_res.branches:
                assert len(mid_res.branches) == 1
                branch = mid_res.branches[0]
                mask = branch.mask
                if mask is not None:
                    logits[sampling_map[seq_id], :] += bias[mask, 0:vocab_size]
                if branch.temperature is not None:
                    self.seq_id_to_sampling_params[seq_id].temperature = max(
                        branch.temperature, EPSILON_TEMP)
        return logits

    def transform_sampler_output(self, output: SamplerOutput) -> SamplerOutput:
        runner = self.runner
        for out in output.outputs:
            for sample in out.samples:
                seq_id = sample.parent_seq_id
                if seq_id not in self.seq_id_to_sampling_idx:
                    continue
                mid_res = runner.mid_status(seq_id)
                if not mid_res:
                    continue
                if mid_res.error or not mid_res.branches:
                    sample.fast_forward_tokens = [runner.eos_token_id]
                else:
                    splice = mid_res.branches[0].find_splice(
                        sample.output_token)
                    if splice is not None:
                        bt = splice.backtrack
                        tokens = splice.ff_tokens[:]
                        if mid_res.branches[0].mask is not None:
                            if bt == 0:
                                tokens.insert(0, sample.output_token)
                            elif bt == 1:
                                bt = 0
                        assert bt == 0, "Backtrack not supported"
                        sample.fast_forward_tokens = tokens[:]
                    else:
                        prob = sample.logprobs[sample.output_token]
                        fixed = False
                        if prob.rank > 1:
                            params = self.seq_id_to_sampling_params[seq_id]
                            # if we didn't sample top-1, and we're supposed to
                            # be greedy (temperature approx 0.0), then fix it
                            if params.temperature <= EPSILON_TEMP:
                                old_tok = sample.output_token
                                for token, logprob in sample.logprobs.items():
                                    if logprob.rank == 1:
                                        sample.output_token = token
                                        fixed = True
                                        break
                                self.log(f"FIXED {old_tok} -> {sample.output_token}")
                        tokens = [sample.output_token]
                        if fixed:
                            sample.fast_forward_tokens = tokens[:]
                    runner.tokens_generated(seq_id, tokens, 0)
        return output
