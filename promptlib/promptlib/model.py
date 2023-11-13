
from .prompt import PromptNode
from .prompt import BeginBlockNode
from .models import LLM, ChatLLM

class ModelNode(PromptNode):

    def __init__(self, llm:LLM, **args):
        super().__init__(**args)
        self.llm = llm

    def generate(self, prefix, max_tokens, **kwargs):
        # this function should be overriden to generate text
        return self.llm(prefix, max_new_tokens=max_tokens, **kwargs)

    def get_text(self):        
        return ""
    
    def _get_plan_step(self):
        dict = {"model": self.llm.get_name()}
        dict.update(self._get_attributes())
        return {"Model": dict}

class ChatModelNode(ModelNode):
    def __init__(self, llm:ChatLLM, **args):
        super().__init__(llm, **args)

    def generate(self, messages, max_tokens, **kwargs):
        # this function should be overriden to generate text
        return self.llm(messages, max_tokens=max_tokens, **kwargs)


def set_model(prompt_code:PromptNode, llm:LLM) -> PromptNode:
    if( isinstance(llm, ChatLLM)):
        model = ChatModelNode(llm)
    else:
        model = ModelNode(llm)
    prompt_code.add_child(model)
    return model

def begin_system(prompt_code:PromptNode, id:str=None, tag:str=None) -> PromptNode:
    return prompt_code.begin_chat(role="system", id=id, tag=tag)

def begin_user(prompt_code:PromptNode, id:str=None, tag:str=None) -> PromptNode:
    return prompt_code.begin_chat(role="user", id=id, tag=tag)

def begin_assistant(prompt_code:PromptNode, id:str=None, tag:str=None) -> PromptNode:
    return prompt_code.begin_chat(role="assistant", id=id, tag=tag)
