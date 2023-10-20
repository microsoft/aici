
class PromptNode:
    """
    A node in a prompt tree, representing a prompt or a group of prompts.
    """

    def __init__(self, id:str=None, tags:dict=None):
        self.children = []
        self.parent = None
        self.id = id
        self.tags = tags 

    def add_child(self, child):
        # this function should be overriden to validate that adding the child is ok
        child.set_parent(self)
        self.children.append(child)
    
    def children(self):
        return self.children
    
    def set_parent(self, parent):
        self.parent = parent
        
    def _get_attributes(self):
        dict = {}
        if self.id is not None:
            dict["id"] = self.id
        if self.tags is not None:
            dict["tags"] = self.tags
        return dict

    def get_text(self):
        # this function should be overriden
        return ""

    def get_all_text(self, stop_at=None):
        if stop_at is not None and self is stop_at:
            return ""
        t = self.get_text()
        if self.parent is not None:
            t = self.parent.get_all_text() + t
        return t

    def get_all_chat_text(self):
        if self.parent is not None:
            return self.parent.get_all_chat_text()

        # only begin blocks will actually create chat text
        return []

    def get_partial_chat_text(self):
        if self.parent is not None:
            partial = self.parent.get_partial_chat_text()
            partial[0]["content"] += self.get_text()

        return None

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
            steps.append(self._get_plan_step())

        if self.children is not None:
            if len(self.children) > 1:
               child_steps = []
               for c in self.children:
                   child_steps.extend(c._get_plan_steps_descending(stopNode=stopNode))
               steps.append({"Parallel": {"steps": [child_steps]}})
            elif len(self.children) == 1:
                steps.extend(self.children[0]._get_plan_steps_descending(stopNode=stopNode)),
        return [s for s in steps if s is not None]
    
    def _get_plan_steps_ascending(self, includeSelf=True):
        if self.parent is None:
            steps = []
        else:
            steps = self.parent._get_plan_steps_ascending()
        if includeSelf:
            steps.append(self._get_plan_step())
        return steps

    def build_linear_plan(self):
        steps = self._get_plan_steps_ascending()
        steps = [s for s in steps if s is not None]
        return {"steps": steps}

    # This builds a plan to execute all the children starting at self
    def build_tree_plan(self):
        steps = {"steps": self._get_plan_steps_descending()}
        return steps


class TextNode(PromptNode):

    def __init__(self, text:str):
        super().__init__()
        self.text = text

    def get_text(self):
        return self.text
    
    def _get_plan_step(self):
        dict = {"text": self.text}
        dict.update(self._get_attributes())
        return {"Fixed": dict}


def append(prompt_code:PromptNode, text:str) -> PromptNode:
    node = TextNode(text)
    prompt_code.add_child(node)
    return node


class BeginBlockNode(PromptNode):

    def __init__(self, **args):
        super().__init__(**args)

    def set_end_block(self, end_block):
        self._end_block = end_block

    def _get_plan_steps_descending(self, stopNode=None, includeSelf=True):
        block_steps = super()._get_plan_steps_descending(stopNode=self._end_block, includeSelf=includeSelf)
        block_dict = {"steps": block_steps}
        block_dict.update(self._get_attributes())
        steps = [{"block":block_dict}]
        steps.extend(self._end_block._get_plan_steps_descending())
        return steps


def begin(prompt_code:PromptNode, id:str=None, tags=None) -> PromptNode:
    node = BeginBlockNode(id=id)
    prompt_code.add_child(node)
    return node


class BeginChatBlockNode(BeginBlockNode):

    def __init__(self, role:str, **args):
        super().__init__(**args)
        self.role = role

    def get_all_chat_text(self):
        if( self.role is None):
            raise Exception("BeginChatBlock must have a role before calling get_all_chat_text")
        
        if( self._end_block is None):
            raise Exception("BeginChatBlock must have a matching EndBlock before calling get_all_chat_text")
        
        block_text = self._end_block.get_all_text(stop_at=self)
        chat_text = self._get_prev_chat_text()
        chat_text.append({"role":self.role, "content":block_text})
        return chat_text

    def _get_prev_chat_text(self):
        prev_begin_block = self.get_parent_of_type(BeginChatBlockNode)
        if prev_begin_block is not None:
            chat_text = prev_begin_block.get_all_chat_text()
        else:
            chat_text = []
        return chat_text        

    def get_partial_chat_text(self):
        assert self.role is not None, "BeginChatBlock must have a role before calling get_partial_chat_text"
        chat_text = self._get_prev_chat_text()
        chat_text.append({"role":self.role, "content":""})
        return chat_text

    def _get_attributes(self):
        return {"role": self.role}


def begin_chat(prompt_code:PromptNode, role:str, id:str=None, tags=None) -> PromptNode:
    node = BeginChatBlockNode(role, id=id, tags=tags)
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
        steps.append({"block":{"steps": block_steps}})
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
