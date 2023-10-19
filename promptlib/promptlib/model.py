
from .prompt import PromptNode
from .prompt import BeginBlockNode
from .models import LLM, ChatLLM

class ModelNode(PromptNode):

    def __init__(self, llm:LLM):
        super().__init__()
        self.llm = llm

    def generate(self, prefix, max_tokens, **kwargs):
        # this function should be overriden to generate text
        return self.llm(prefix, max_new_tokens=max_tokens, **kwargs)

    def get_text(self):        
        return ""

class ChatModelNode(ModelNode):
    def __init__(self, llm:ChatLLM):
        super().__init__(llm)

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

def begin_system(prompt_code:PromptNode, id:str=None, tags=None) -> PromptNode:
    return prompt_code.begin_chat(role="system", id=id, tags=tags)

def begin_user(prompt_code:PromptNode, id:str=None, tags=None) -> PromptNode:
    return prompt_code.begin_chat(role="user", id=id, tags=tags)

def begin_assistant(prompt_code:PromptNode, id:str=None, tags=None) -> PromptNode:
    return prompt_code.begin_chat(role="assistant", id=id, tags=tags)
