from pydantic import BaseModel
from typing import List, Optional, Union
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

EPSILON_TEMP = 1.5e-5

class RunRequest(BaseModel):
    model: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[ChatCompletionMessageParam]] = None
    controller: str
    controller_arg: Union[str, dict]
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: Optional[int] = None

    def to_sampling_params(self):
        r = SamplingParams(
            temperature=max(self.temperature, EPSILON_TEMP),
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            ignore_eos=False,
            logprobs=10,
        )
        r.has_aici = True # type: ignore
        return r


class SetTagsRequest(BaseModel):
    module_id: str
    tags: List[str]