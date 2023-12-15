
from .prompt import PromptNode, append, begin, begin_chat, end, PromptProgram
from .gen import gen, choose, wait
from .model import set_model, begin_assistant, begin_user, begin_system
from .constrain import constrain

from .models import LLM, TransformersLLM

from .aici import AICI

setattr(PromptNode, "append", append)
setattr(PromptNode, "begin", begin)
setattr(PromptNode, "begin_chat", begin_chat)
setattr(PromptNode, "begin_system", begin_system)
setattr(PromptNode, "begin_assistant", begin_assistant)
setattr(PromptNode, "begin_user", begin_user)
setattr(PromptNode, "end", end)
setattr(PromptNode, "gen", gen)
setattr(PromptNode, "choose", choose)
setattr(PromptNode, "wait", wait)
setattr(PromptNode, "constrain", constrain)
setattr(PromptNode, "set_model", set_model)
