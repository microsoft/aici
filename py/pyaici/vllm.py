from typing import List, Union, Dict, Any, Tuple, cast

import torch

from vllm import LLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata, SequenceGroup, SequenceStatus, Sequence
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.utils import Counter
from vllm.core.block_manager import BlockSpaceManager

from .comms import AiciRunner, BenchTimer


def install(runner: AiciRunner):
    timer = BenchTimer("initiate_step")
    bias_timer = BenchTimer("bias")
    finish_timer = BenchTimer("finish")

    def initiate_step(
        scheduler: Scheduler,
        counter: Counter,
        scheduler_outputs: SchedulerOutputs,
    ):
        with timer:
            return do_initiate_step(scheduler, counter, scheduler_outputs)

    def do_initiate_step(
        scheduler: Scheduler,
        counter: Counter,
        scheduler_outputs: SchedulerOutputs,
    ):
        runner.flush_logit_bias()

        for f in scheduler.freed_seq_ids:
            runner.step_free_seq(f)

        max_context_len = 0
        num_gen = 0

        steps: List[tuple[SequenceGroup, Sequence]] = []
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            ff_seqs = [
                seq for seq in seqs if seq.data.num_pending_ff_tokens > 0
            ]
            is_ff = len(ff_seqs) > 0
            if is_ff:
                # print("FF", [(seq.seq_id, seq.data.num_pending_ff_tokens, seq.skip_round) for seq in seqs])
                assert scheduler_outputs.prompt_run
                seqs = ff_seqs
            elif scheduler_outputs.prompt_run:
                assert len(seqs) == 1
            for seq in seqs:
                steps.append((seq_group, seq))
                id = seq.seq_id
                max_context_len = max(max_context_len, seq.get_len())

                if seq.skip_round:
                    seq.skip_round = False
                    num_gen += 1
                    runner.step_add_pre(id)
                elif seq.data.num_pending_ff_tokens:
                    runner.step_add_pre(id)
                elif scheduler_outputs.prompt_run:
                    runner.step_add_pre(id, req_id=seq_group.request_id)
                else:
                    num_gen += 1
                    runner.step_add_pre(id)

        if num_gen == 0:
            runner.disable_attn_mask = True

        fork_map, suspend_ids = runner.step_finish_pre(max_context_len)
        if fork_map is None:
            return
        assert suspend_ids is not None
        used = [False for _ in steps]

        for _op_idx, parent_idx in enumerate(fork_map):
            seq_group, seq = steps[parent_idx]
            clone_id = None
            if used[parent_idx]:
                assert not seq.is_finished()
                copy = seq.fork(next(counter))
                seq_group.add(copy)
                seq_group.sampling_params.dynamic_forks = True
                scheduler.fork_seq(seq, copy)
                clone_id = seq.seq_id
                seq = copy
            else:
                used[parent_idx] = True
            runner.step_add_mid(seq.seq_id, clone_id=clone_id)

        for id in suspend_ids:
            seq_group, seq = steps[id]
            assert not used[id]
            # print("SUSP", seq.seq_id)
            used[id] = True
            seq.skip_round = True

        runner.step_finish_mid()

        for idx in range(len(steps)):
            if used[idx]:
                continue
            seq_group, seq = steps[idx]
            seq.status = SequenceStatus.FINISHED_ABORTED
            seq_group.remove(seq.seq_id)
            scheduler.free_seq(seq)

    def apply_dynamic_logit_bias(logits: torch.Tensor):
        bias = (torch.from_numpy(runner.recv_logit_bias()).to(
            logits.device).to(logits.dtype))
        logits += bias

    def recv_attention_mask():
        with bias_timer:
            return torch.from_numpy(runner.recv_attention_mask())

    def append_ff_tokens(
        llm_engine: LLMEngine,
        _seq_group: SequenceGroup,
        child_seqs: List[Tuple[Sequence, Sequence]],
    ):
        runner.recent_seqs = {}
        for seq, parent in child_seqs:
            assert not seq.skip_round
            runner.recent_seqs[seq.seq_id] = seq
            # lookup by parent - the child wasn't born yet when response was generated
            resp = runner.response_by_seq_id(parent.seq_id).get(
                "result", None) or {}
            backtrack: int = resp.get("backtrack", 0)
            ff: List[int] = resp.get("ff_tokens", []).copy()
            if backtrack:
                assert seq is parent
                seq.backtrack(backtrack)
                llm_engine.scheduler.block_manager.trim_physical_blocks(seq)
                assert ff
                t = ff.pop(0)
                seq.append_token_id(t, {t: 0.0})
            elif ff:
                t = ff.pop(0)
                assert t == seq.data.output_token_ids[-1], (
                    "FF", t, seq.data.output_token_ids, ff)
            last_tok = seq.data.output_token_ids[-1]
            # replace sampled EOS with space - at least Llama models get confused by EOS
            if (not backtrack and not ff
                    and last_tok == llm_engine.tokenizer.eos_token_id):
                if runner.space_token == -1:
                    sp = llm_engine.tokenizer.tokenize(" ")[-1]
                    runner.space_token = cast(
                        int, llm_engine.tokenizer.convert_tokens_to_ids(sp))
                # note that we keep last_tok as EOS, to pass to the post_process()
                seq.data.output_token_ids[-1] = runner.space_token
            toks = [last_tok]
            if ff:
                # first, decode with only one token
                llm_engine._decode_sequence(seq)
                # print("FF", seq.seq_id, ff, resp)
                for t in ff:
                    # probability of the token is 1.0, so logprob is 0.0
                    seq.append_token_id(t, {t: 0.0})
                seq.data.num_pending_ff_tokens = len(ff) + 1
                toks += ff
            # print("TTT", backtrack, seq.data.output_token_ids)
            clone_id = None
            if parent is not seq:
                clone_id = parent.seq_id
            runner.step_add_post(seq.seq_id, backtrack, toks, clone_id)

    def finish_sampling():
        with finish_timer:
            for seq_id in runner.step_finish_post():
                seq: Sequence = runner.recent_seqs[seq_id]
                # print("FINISH", seq_id, seq.data.output_token_ids)
                seq.status = SequenceStatus.FINISHED_STOPPED
        runner.recent_seqs = {}

    SamplingParams.apply_dynamic_logit_bias = apply_dynamic_logit_bias
    SamplingParams.initiate_step = initiate_step
    SamplingParams.finish_sampling = finish_sampling
    SamplingParams.append_ff_tokens = append_ff_tokens
    SamplingParams.recv_attention_mask = recv_attention_mask  # type: ignore
