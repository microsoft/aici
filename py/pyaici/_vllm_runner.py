from typing import (List, Optional)
from dataclasses import dataclass
from fastapi import Request

from vllm.utils import random_uuid
from vllm.config import ModelConfig
from vllm.sampling_params import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_engine import (LoRAModulePath,
                                                    OpenAIServing)
from vllm.lora.request import LoRARequest

from ._vllm_protocol import RunRequest
from ._vllm_sampling_ctrl import AiciSamplingController
from .comms import AiciRunner

# TODO catch ValueError and use self.create_streaming_error_response(str(e))


@dataclass
class ReqInfo:
    request_id: str
    prompt: List[int]
    sampling_params: SamplingParams
    error: Optional[dict] = None
    lora_request: Optional[LoRARequest] = None


class AiciRunnerCompletion(OpenAIServing):

    def __init__(self, aici_runner: AiciRunner, engine: AsyncLLMEngine,
                 model_config: ModelConfig, served_model_names: List[str],
                 lora_modules: Optional[List[LoRAModulePath]]):
        super().__init__(engine=engine,
                         model_config=model_config,
                         served_model_names=served_model_names,
                         lora_modules=lora_modules)
        self.aici_runner = aici_runner
        self.sampling_controller = AiciSamplingController(aici_runner)
        self.empty_prompt: List[int] = self.tokenizer("").input_ids
        if not self.empty_prompt:
            # if there's no start symbol, add a space, otherwise Engine
            # gets stuck on empty prompt
            self.empty_prompt = self.tokenizer(" ").input_ids
            assert self.empty_prompt

    # this is separate from create_completion() so fastapi exceptions
    # from .instantiate_async() are properly sent to the user
    async def prep_completion(self, request: RunRequest):
        if request.model:
            error_check_ret = await self._check_model(request)
            if error_check_ret is not None:
                return error_check_ret
        else:
            request.model = self.served_model_names[0]
        prompt = self.tokenizer(request.prompt).input_ids
        req_info = ReqInfo(
            request_id=f"run-{random_uuid()}",
            prompt=[],
            sampling_params=request.to_sampling_params(),
        )
        req_info.sampling_params.stop_token_ids = []
        inst_res = await self.aici_runner.instantiate_async(
            req_info.request_id,
            prompt,
            request.controller,
            request.controller_arg,
        )

        if isinstance(inst_res, dict):
            req_info.error = inst_res
        else:
            assert isinstance(inst_res, list)
            # Engine doesn't like prompts with no tokens
            # self.empty_prompt is either start symbol or a single space
            if len(inst_res) == 0:
                inst_res = self.empty_prompt
            req_info.prompt = inst_res
            req_info.lora_request = self._maybe_get_lora(request)

        return req_info

    async def create_completion(self, req_info: ReqInfo, raw_request: Request):
        """Completion API for AICI controllers.

        See https://github.com/microsoft/aici/blob/main/docs/REST.md
        """
        runner = self.aici_runner
        yield runner.data_line(
            runner.initial_json(req_info.request_id,
                                self.served_model_names[0]))

        if req_info.error:
            # error case
            yield runner.data_line(req_info.error)
            yield runner.final_data()
            return

        generator = self.engine.generate(
            {"prompt_token_ids": req_info.prompt},
            request_id=req_info.request_id,
            sampling_params=req_info.sampling_params,
            lora_request=req_info.lora_request)

        previous_texts: List[str] = []
        ff_tokens = len(req_info.prompt)
        sampled_tokens = 0
        seq_id: Optional[int] = None
        last_finish_reason = ""
        prev_token_ids = 0
        ff_tokens -= 1

        async for res in generator:
            if seq_id is None:
                seq_id = self.sampling_controller.resolve_req_id(
                    req_info.request_id)
                assert seq_id is not None

            # Abort the request if the client disconnects.
            if await raw_request.is_disconnected():
                await self.engine.abort(req_info.request_id)
                runner.seq_freed(seq_id)
                raise StopAsyncIteration()

            # TODO simplify this - there is only one fork
            forks = []
            for output in res.outputs:
                curr_len = len(output.token_ids)
                ff_tokens += max(1, curr_len - prev_token_ids)
                prev_token_ids = curr_len
                sampled_tokens += 1

                i = output.index
                while len(previous_texts) <= i:
                    previous_texts.append("")
                delta_text = output.text[len(previous_texts[i]):]
                previous_texts[i] = output.text

                last_finish_reason = output.finish_reason
                fork_res = runner.seq_logs(
                    seq_id,
                    index=i,
                    text=delta_text,
                    finish_reason=last_finish_reason,
                )
                forks.append(fork_res)
            yield runner.data_line(
                runner.run_json(forks,
                                runner.usage_json(ff_tokens, sampled_tokens)))

        if seq_id is not None:
            runner.seq_freed(seq_id)
            if seq_id in runner.pending_generated_tokens:
                self.sampling_controller.log("waiting for step finish")
                await runner.wait_for_step_finish()
                fork_res = runner.seq_logs(
                    seq_id,
                    index=0,
                    text="",
                    finish_reason=last_finish_reason,
                )
                yield runner.data_line(
                    runner.run_json([fork_res],
                                    runner.usage_json(ff_tokens,
                                                      sampled_tokens)))

        yield runner.final_data()
