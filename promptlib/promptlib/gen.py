from .prompt import PromptNode
from .model import ModelNode, ChatModelNode
from .endpoint import EndpointNode
from .constrain import ConstraintNode

class GenNode(PromptNode):

    def __init__(self, max_tokens=1000, id:str=None, tag:str=None, ignore:list=None, **genargs):
        args = {}
        if id is not None:
            args["id"] = id
        if tag is not None:
            args["tag"] = tag

        super().__init__(**args)
        self.max_tokens = max_tokens
        self.generated_text = None
        self.genargs = genargs
        self.ignore = ignore
    
    def set_parent(self, parent): 
        super().set_parent(parent)
        model = self.get_parent_of_type(ModelNode)
        endpoint = self.get_parent_of_type(EndpointNode)

        # TODO this will fail in cases where we are building subprompts and then only later prepending a model
        assert (model is not None) or (endpoint is not None), "Gen must have an ancestor of type ModelNode or EndpointNode"
        if model is not None:
            self.is_chat = isinstance(model, ChatModelNode)

    def _generate_text(self, prefix):
        model = self.get_parent_of_type(ModelNode)
        constraint_nodes = self.get_all_ancestors_of_type(ConstraintNode, stop_at_type=ModelNode)
        constraints = [c.constraint for c in constraint_nodes]
        if model is None:
            raise Exception("Gen must be a child of Model")
        return model.generate(prefix, self.max_tokens, constraints=constraints, **self.genargs)

    def get_text(self):
        assert self.parent is not None, "Gen must have a parent"

        if self.generated_text is None:            
            if self.is_chat:            
                prefix = self.parent.get_partial_chat_text()
                # prefix is a list of dicts with keys "content" and "role"
            else:
                prefix = self.parent.get_all_text()
                # prefix is a string
        
            self.generated_text = self._generate_text(prefix)

        return self.generated_text

    def _get_plan_step(self):
        dict = {"max_tokens": self.max_tokens}
        dict.update(self._get_attributes())
        return {"Gen": dict}
    
    def _get_attributes(self):
        attr = super()._get_attributes()
        attr.update({"genargs": self.genargs})
        if self.ignore is not None:
            attr.update({"mask_tags": self.ignore})
        return attr


def gen(prompt_code:PromptNode, max_tokens=1000, **genargs) -> PromptNode:
    node = GenNode(max_tokens, **genargs)
    prompt_code.add_child(node)
    return node


class ChoiceNode(GenNode):
    
    def __init__(self, choices:list, **args):
        self.choices = choices
        #self.choices_logprobs = None figure out how to get logprobs
        self.choice = None
        super().__init__(**args) # TODO add genargs to super that end up logit_biasing to the choices

    def _get_plan_step(self):
        dict = {"rx": r"|".join(self.choices)}
        dict.update(self._get_attributes())
        return {"Gen": dict}


def choose(prompt_code:PromptNode, choices:list) -> PromptNode:
    node = ChoiceNode(choices)
    prompt_code.add_child(node)
    return node
