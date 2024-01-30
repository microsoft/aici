import re
from typing import List
from .aici import AICI

from pyaici import ast

class PromptNode:
    """
    A node in a prompt tree, representing a prompt or a group of prompts.
    """

    id_counter = 0

    def __init__(self, attrs:List[str]=None):
        self.children = []
        self.parent = None
        self.id = self.get_next_id()
        self.attrs = attrs

    @classmethod
    def get_next_id(cls):
        cls.id_counter += 1
        return f"_{cls.id_counter}"

    def add_child(self, child):
        # this function should be overriden to validate that adding the child is ok
        child.set_parent(self)
        self.children.append(child)
    
    def children(self):
        return self.children
    
    def set_parent(self, parent):
        self.parent = parent

    def get_parent_of_type(self, type):
        if self.parent is None:
            return None
        if isinstance(self.parent, type):
            return self.parent
        return self.parent.get_parent_of_type(type)

    def get_all_ancestors_of_type(self, type, stop_at_type=None):
        ret = []
        if self.parent is not None and (stop_at_type is None or not isinstance(self, stop_at_type)):
            if isinstance(self.parent, type):
                ret.append(self.parent)
            ret.extend(self.parent.get_all_ancestors_of_type(type))

        return ret

    def _get_plan_step(self):
        # this function should be overriden
        return None

    def _get_plan_steps_descending(self, stopNode=None, includeSelf=True):
        if self is stopNode:
            return []
        
        steps = []
        if includeSelf:
            plan_step = self._get_plan_step()
            if( plan_step is None):
                pass
            elif( isinstance(plan_step, list)):
                steps.extend(plan_step)
            else:
                steps.append(plan_step)

        if self.children is not None:
            if len(self.children) > 1:
               child_branches = []
               for c in self.children:
                   child_branches.append(c._get_plan_steps_descending(stopNode=stopNode))
               steps.append({"Fork": {"branches": child_branches}})
            elif len(self.children) == 1:
                steps.extend(self.children[0]._get_plan_steps_descending(stopNode=stopNode)),
        return [s for s in steps if s is not None]
    
    def _get_match_content(self):
        return "{{" + self.id + "}}"

    def _get_predecessor_matches(self, attrs:list):
        id_match = [] # list of tuples (id, bool)
        if self.parent is not None:
            id_match.extend(self.parent._get_predecessor_matches(attrs))

        match_content = self._get_match_content()
        # match if the intersection of attrs and self.attrs is non-empty
        if self.attrs is not None and attrs is not None:
            id_match.append((self.id, len(set(self.attrs).intersection(attrs)) > 0, match_content))
        else:
            id_match.append((self.id, False, match_content))
        return id_match

    # This builds a plan to execute all the children starting at self
    def build_tree_plan(self):
        steps = {"steps": self._get_plan_steps_descending()}
        return steps

    def __add__(self, o):
        assert isinstance(o, str), "Can only add strings to a prompt"
        return append(self, o)


class PromptProgram(PromptNode):
    def __init__(self, endpoint:AICI):
        super().__init__()
        self.endpoint = endpoint

    def run(self):
        plan = self.build_tree_plan()
        return self.endpoint.run(plan)

class TextNode(PromptNode):

    def __init__(self, text:str, attrs:List[str]=None):
        super().__init__(attrs=attrs)
        self.text = text

    def get_text(self):
        return self.text
    
    def _get_match_content(self):
        return self.get_text()
    
    def _get_plan_step(self):
        varnames = re.findall(r'{{(.*?)}}', self.text)
        if len(varnames) > 0:
            return [
                ast.wait_vars(*varnames),
                ast.label(self.id, ast.fixed(self.text, expand_vars=True, tag=self.id))
            ]
        else:
            return [
                ast.label(self.id,ast.fixed(self.text, expand_vars=False, tag=self.id))
            ]


def append(prompt_code:PromptNode, text:str, attrs:List[str]=None) -> PromptNode:
    node = TextNode(text, attrs=attrs)
    prompt_code.add_child(node)
    return node


class BeginBlockNode(PromptNode):

    def __init__(self, hidden=False, **args):
        super().__init__(**args)
        self.hidden = hidden

    def set_end_block(self, end_block):
        self._end_block = end_block

    def _get_plan_steps_descending(self, stopNode=None, includeSelf=True):
        steps = super()._get_plan_steps_descending(stopNode=self._end_block, includeSelf=includeSelf)

        if self.hidden:
            steps[0]["label"] = self.id
            steps.extend(ast.fixed("", following=self.id))

        steps.extend(self._end_block._get_plan_steps_descending(stopNode=stopNode, includeSelf=True))
        return steps


def begin(prompt_code:PromptNode, hidden=False) -> PromptNode:
    node = BeginBlockNode(hidden=hidden)
    prompt_code.add_child(node)
    return node


class EndBlockNode(PromptNode):

    def __init__(self, **args):
        super().__init__(**args)
        self.begin_block = None

    def set_parent(self, parent):
        if isinstance(parent, BeginBlockNode):
            self.begin_block = parent
        else:
            self.begin_block = parent.get_parent_of_type(BeginBlockNode)

        if self.begin_block is None:
            raise Exception("EndBlock must be a child of BeginBlock")

        self.begin_block.set_end_block(self)

        super().set_parent(parent)

    def get_parent_of_type(self, type):
        if self.begin_block is None:
            return None
        if isinstance(self.begin_block, type):
            return self.begin_block
        return self.begin_block.get_parent_of_type(type)
    
    def _get_plan_steps_ascending(self):
        steps = self.begin_block._get_plan_steps_ascending(includeSelf=False)
        block_steps = self.begin_block._get_plan_steps_descending(stopNode=self)    
        steps.append(block_steps)
        return steps
        


def end(prompt_code:PromptNode) -> PromptNode:
    node = EndBlockNode()
    prompt_code.add_child(node)
    return node


## Other nodes to add:
##   - reference (returns the value of another id) -- also consider regex applied to all searches to extract stuff
##   - loops
##   - if/else
##   - functions (call a lambda python function)
##   - constrain (regex, cfg, etc)
