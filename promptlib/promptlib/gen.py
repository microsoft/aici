from .prompt import PromptNode
from .model import ModelNode, ChatModelNode
from .endpoint import EndpointNode
from .constrain import ConstraintNode

from typing import List

class GenNode(PromptNode):

    def __init__(self, max_tokens=1000, attrs:List[str]=None, ignore:list=None, set_var=None, append_var=None, **genargs):
        args = {}
        if attrs is not None:
            args["attrs"] = attrs

        super().__init__(**args)
        self.max_tokens = max_tokens
        self.generated_text = None
        self.genargs = genargs
        self.ignore = ignore
        self.set_var = set_var
        self.append_var = append_var
    
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

        id_ignore = None
        idx = None
        if self.ignore is not None:
            # figure out which past commands are in the ignore list
            id_ignore = self.parent._get_predecessor_matches(self.ignore)
            # find the index of the first True in the list
            idx = _find_first_match_index(id_ignore, lambda x: x[1] == True)

        # if nothing needs to be ignored, just continue as normal
        if idx is None:
            return {"Gen": dict}

        # if we need to ignore stuff, then we need to do some extra work
        follows = id_ignore[idx][0] # follows right *before* idx
        fixed_text_parts = []
        for i in range(idx, len(id_ignore)):
            if id_ignore[i][1] == False:
                # if id is a gen:
                fixed_text_parts.append({"String": {"str": id_ignore[i][2]}})

        fixed_text = {"Concat": {"parts": fixed_text_parts }}
        if( self.set_var is None):
            self.set_var = "gen_var" + self.id

        dict.update({"stmts": [{"Set": {"var": self.set_var, "expr": {"Current": {}}}}]})

        fixed_dict = {"following": follows, "text": fixed_text}
        ## basic outline:
        ##    the main branch will wait for variables to be generated and then bring that back into the main branch
        return {"Fork": {"branches": [
            # fork a branch that will backtrack to ignore selected sections of text and generated text
            [{"Fixed": fixed_dict}, # the fixed_dict includes: 
                                    #  (1) an instruction to backtrack (follow) the prompt right before the first ignored text;
                                    #  (2) all the text (either fixed text or by variable reference) that was added or generated but not ignored
             {"Gen": dict}],        # Note that the "Gen" saves its value in self.set_var
            # main branch waits for generation to complete and then brings the generated value back into the main branch
            [{"Wait": {"vars": [self.set_var]}}, 
             {"Fixed": {"text": {"Var": {"var": self.set_var}}}}]
        ]}}

    
    def _get_attributes(self):
        attr = super()._get_attributes()
        attr.update(self.genargs)
        #if self.ignore is not None:
        #    attr.update({"mask_tags": self.ignore})
        if self.append_var is not None:
            attr.update({"append_to_var": self.append_var})
        if self.set_var is not None:
            attr.update({"set_var": self.set_var})
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
        dict = {"options": self.choices}
        dict.update(self._get_attributes())
        return {"Choose": dict}


def choose(prompt_code:PromptNode, choices:list) -> PromptNode:
    node = ChoiceNode(choices)
    prompt_code.add_child(node)
    return node


class WaitNode(PromptNode):

    def __init__(self, vars:list, **args):
        super().__init__(**args)
        self.vars = vars

    def _get_plan_step(self):
        dict = {"vars": self.vars}
        dict.update(self._get_attributes())
        return {"Wait": dict}
    
def wait(prompt_code:PromptNode, vars:list) -> PromptNode:
    node = WaitNode(vars)
    prompt_code.add_child(node)
    return node




def _find_first_match_index(list, predicate):
    return next((i for i, x in enumerate(list) if predicate(x)), None)
