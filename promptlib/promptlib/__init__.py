
from .prompt import PromptNode, append, begin, end, PromptProgram
from .gen import gen, choose, wait

from .aici import AICI

setattr(PromptNode, "append", append)
setattr(PromptNode, "begin", begin)
setattr(PromptNode, "end", end)
setattr(PromptNode, "gen", gen)
setattr(PromptNode, "choose", choose)
setattr(PromptNode, "wait", wait)

