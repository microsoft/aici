from typing import List, Union, Dict, Any

import torch

from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceGroup, SequenceStatus, Sequence
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.utils import Counter

from .comms import AiciRunner


def install(runner: AiciRunner):
    def initiate_step(
        scheduler: Scheduler,
        counter: Counter,
        scheduler_outputs: SchedulerOutputs,
    ):
        runner.flush_logit_bias()

        for f in scheduler.freed_seq_ids:
            runner.step_free_seq(f)

        max_context_len = 0
        num_gen = 0

        steps: list[tuple[SequenceGroup, Sequence]] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            ff_seqs = [seq for seq in seqs if seq.data.num_pending_ff_tokens > 0]
            is_ff = len(ff_seqs) > 0
            if is_ff:
                assert scheduler_outputs.prompt_run
                seqs = ff_seqs
            elif scheduler_outputs.prompt_run:
                assert len(seqs) == 1
            for seq in seqs:
                steps.append((seq_group, seq))
                id = seq.seq_id
                if seq.data.num_pending_ff_tokens:
                    toks = seq.get_token_ids()
                    max_context_len = max(max_context_len, len(toks))
                    runner.step_add_tokens(
                        id,
                        toks[-seq.data.num_pending_ff_tokens :],
                        clone_id=seq.data.parent_id,
                    )
                    seq.data.parent_id = None
                elif scheduler_outputs.prompt_run:
                    toks = seq.get_token_ids()
                    max_context_len = max(max_context_len, len(toks))
                    runner.step_add_prompt(
                        id,
                        prompt=toks,
                        req_id=seq_group.request_id,
                    )
                else:
                    num_gen += 1
                    out = seq.data.output_token_ids
                    max_context_len = max(max_context_len, seq.get_len())
                    runner.step_add_tokens(
                        id, tokens=[out[-1]], clone_id=seq.data.parent_id
                    )
                    seq.data.parent_id = None

        if num_gen == 0:
            runner.disable_attn_mask = True

        sent = runner.step_finish(max_context_len)
        if not sent:
            return

        fork_map = runner.process_forks()
        used = [False for _ in steps]

        for _op_idx, parent_idx in enumerate(fork_map):
            seq_group, seq = steps[parent_idx]
            clone_id = None
            if used[parent_idx]:
                assert not seq.is_finished()
                copy = seq.fork(next(counter))
                seq_group.add(copy)
                scheduler.fork_seq(seq, copy)
                clone_id = seq.seq_id
                copy.data.parent_id = None # don't clone it again in the next step
                seq = copy
            else:
                used[parent_idx] = True
            runner.step_add_tokens(seq.seq_id, tokens=[], clone_id=clone_id)

        runner.step_finish2()

        for idx in range(len(steps)):
            if used[idx]:
                continue
            seq_group, seq = steps[idx]
            seq.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(seq.seq_id)
            scheduler.free_seq(seq)

    def apply_dynamic_logit_bias(logits: torch.Tensor):
        bias = (
            torch.from_numpy(runner.recv_logit_bias())
            .to(logits.device)
            .to(logits.dtype)
        )
        logits += bias

    def recv_attention_mask():
        return torch.from_numpy(runner.recv_attention_mask())

    def append_ff_tokens(seq_group: SequenceGroup):
        for seq in seq_group.get_seqs():
            resp = runner.response_by_seq_id(seq.seq_id)
            ff = resp and resp.get("ff_tokens", None)
            if ff:
                # print("FF", seq.seq_id, ff, resp)
                seq.pending_ff_tokens = ff

    SamplingParams.apply_dynamic_logit_bias = apply_dynamic_logit_bias
    SamplingParams.initiate_step = initiate_step
    SamplingParams.append_ff_tokens = append_ff_tokens
    SamplingParams.recv_attention_mask = recv_attention_mask
