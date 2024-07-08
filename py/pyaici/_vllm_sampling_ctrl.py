import torch

from typing import List, Dict, Optional

from vllm.sequence import SamplingController, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .comms import AiciRunner


class AiciSamplingController(SamplingController):

    def __init__(self, aici_runner: AiciRunner):
        super().__init__()
        self.runner = aici_runner
        self.logit_pending = False
        self.seq_id_to_sampling_idx: Dict[int, int] = {}
        self.req_id_to_seq_id: Dict[str, int] = {}

    def resolve_req_id(self, req_id: str) -> Optional[int]:
        return self.req_id_to_seq_id.get(req_id)

    def prepare(self, sampling_metadata: "SamplingMetadata"):
        runner = self.runner
        seq_id_to_sampling_idx: Dict[int, int] = {}
        req_id_to_seq_id: Dict[str, int] = {}
        for group in sampling_metadata.seq_groups:
            if not getattr(group.sampling_params, "has_aici", False):
                continue
            assert len(group.seq_ids) == 1
            seq_id = group.seq_ids[0]
            req_id_to_seq_id[group.request_id] = seq_id
            sample_indices = group.sample_indices
            if sample_indices:
                assert len(sample_indices) == 1
                runner.assign_seq_id(group.request_id, seq_id, optional=True)
                runner.add_mid(seq_id)
                seq_id_to_sampling_idx[seq_id] = sample_indices[0]
            else:
                pass
        if runner.needs_exec_mid():
            runner.exec_mid()
            self.seq_id_to_sampling_idx = seq_id_to_sampling_idx
            self.req_id_to_seq_id = req_id_to_seq_id
            self.logit_pending = True

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        num_tokens, vocab_size = logits.shape
        if not self.logit_pending:
            return logits
        resp, bias = self.runner.recv_logit_bias_torch()
        sampling_map = self.seq_id_to_sampling_idx
        num_masks, vocab_size2 = bias.shape
        assert num_masks <= num_tokens  # ?
        assert vocab_size2 == vocab_size
        for idx, mid_res in resp.items():
            if mid_res.branches:
                assert len(mid_res.branches) == 1
                mask = mid_res.branches[0].mask
                if mask is not None:
                    logits[sampling_map[idx], :] += bias[mask, :]
        return logits

    def transform_sampler_output(self, output: SamplerOutput) -> SamplerOutput:
        runner = self.runner
        for out in output.outputs:
            for sample in out.samples:
                seq_id = sample.parent_seq_id
                mid_res = runner.mid_status(seq_id)
                if not mid_res:
                    continue
                if mid_res.error or not mid_res.branches:
                    sample.output_token = runner.eos_token_id
                else:
                    splice = mid_res.branches[0].find_splice(
                        sample.output_token)
                    if splice is not None:
                        assert splice.backtrack == 0, "Backtrack not supported"
                        tokens = splice.ff_tokens[:]
                        if len(tokens) == 1:
                            sample.output_token = tokens[0]
                        else:
                            sample.fast_forward_tokens = tokens[:]
                    else:
                        tokens = [sample.output_token]
                    runner.tokens_generated(seq_id, tokens, 0)
        return output
