import torch

from typing import List, Dict, Any, Optional

from vllm.sequence import SamplingController, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

from .comms import AiciRunner


class AiciSamplingController(SamplingController):

    def __init__(self, aici_runner: AiciRunner, vocab_size: int):
        super().__init__()
        self.runner = aici_runner
        self.pending_args: Dict[int, Dict] = {}
        self.freed_ids: List[int] = []
        self.logit_pending = False
        self.vocab_size = vocab_size
        self.aici_id_to_sampling_idx: Dict[int, int] = {}
        self.seq_id_to_aici_id: Dict[int, int] = {}

    def free_seq(self, aici_id: int):
        self.freed_ids.append(aici_id)
        del self.pending_args[aici_id]

    def prepare(self, sampling_metadata: "SamplingMetadata"):
        runner = self.runner
        req: List[Dict] = []
        aici_id_to_sampling_idx: Dict[int, int] = {}
        seq_id_to_aici_id: Dict[int, int] = {}
        pending_args = self.pending_args
        for group in sampling_metadata.seq_groups:
            aici_id = getattr(group.sampling_params, "aici_id", None)
            if not aici_id:
                continue
            assert len(group.seq_ids) == 1
            sample_indices = group.sample_indices
            if sample_indices:
                assert len(sample_indices) == 1
                if aici_id in pending_args:
                    arg = pending_args.pop(aici_id)
                else:
                    arg = {"id": aici_id, "backtrack": 0, "tokens": []}
                seq_id_to_aici_id[group.seq_ids[0]] = aici_id
                aici_id_to_sampling_idx[aici_id] = sample_indices[0]
                req.append(arg)
            else:
                pass
        if req:
            runner.exec_mid_combined(self.vocab_size, req, self.freed_ids)
            self.aici_id_to_sampling_idx = aici_id_to_sampling_idx
            self.seq_id_to_aici_id = seq_id_to_aici_id
            self.freed_ids = []
            self.logit_pending = True

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        num_tokens, vocab_size = logits.shape
        assert vocab_size == self.vocab_size
        if not self.logit_pending:
            return logits
        resp, bias = self.runner.recv_logit_bias_torch()
        sampling_map = self.aici_id_to_sampling_idx
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
        mapping = self.seq_id_to_aici_id
        runner = self.runner
        for out in output.outputs:
            for sample in out.samples:
                seq_id = sample.parent_seq_id
                if seq_id not in mapping:
                    continue
                aici_id = mapping[seq_id]
                mid_res = runner.mid_status(aici_id)
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
                    self.pending_args[aici_id] = {
                        "id": aici_id,
                        "backtrack": 0,
                        "tokens": tokens
                    }
        return output
