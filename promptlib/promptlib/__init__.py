
from promptlib.prompt import PromptNode, append, begin, begin_chat, end
from promptlib.gen import gen, choose
from promptlib.model import set_model, begin_assistant, begin_user, begin_system
from promptlib.constrain import constrain

from promptlib.models import LLM, TransformersLLM

from promptlib.endpoints import AICI, Endpoint
from promptlib.endpoint import EndpointNode, set_endpoint

setattr(PromptNode, "append", append)
setattr(PromptNode, "begin", begin)
setattr(PromptNode, "begin_chat", begin_chat)
setattr(PromptNode, "begin_system", begin_system)
setattr(PromptNode, "begin_assistant", begin_assistant)
setattr(PromptNode, "begin_user", begin_user)
setattr(PromptNode, "end", end)
setattr(PromptNode, "gen", gen)
setattr(PromptNode, "choose", choose)
setattr(PromptNode, "constrain", constrain)
setattr(PromptNode, "set_model", set_model)
setattr(PromptNode, "set_endpoint", set_endpoint)