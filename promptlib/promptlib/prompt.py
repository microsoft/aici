
class PromptNode:
    """
    A node in a prompt tree, representing a prompt or a group of prompts.
    """

    def __init__(self):
        self.children = []
        self.parent = None

    def add_child(self, child):
        # this function should be overriden to validate that adding the child is ok
        child.set_parent(self)
        self.children.append(child)
    
    def children(self):
        return self.children
    
    def set_parent(self, parent):
        self.parent = parent
        
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

    def _get_plan_steps_descending(self):
        steps = [self._get_plan_step()]
        if self.children is not None:
            if len(self.children) > 1:
               # TODO create a branch node
               raise Exception("TODO create a branch node")
            else:
                steps.extend(self.children[0].get_plan_steps_descending())
        return steps
    
    def _get_plan_steps_ascending(self):
        if self.parent is None:
            steps = []
        else:
            steps = self.parent._get_plan_steps_ascending()
        steps.append(self._get_plan_step())
        return steps

    def build_linear_plan(self):
        steps = self._get_plan_steps_ascending()
        # TODO check that this renders correctly
        str_steps = "{steps: ["
        first = True
        for s in steps:
            if s is None:
                continue
            if not first:
                str_steps += ",\n"
            else:
                first = False
            str_steps += s
        str_steps += "]}"
        return str_steps

    # This starts at a root, and builds a plan to execute all the children.
    def build_tree_plan(self):
        # TODO - is it valid to call this function on an interior node?  
        # TODO - is it valid to call this function when there are multiple models?        
        pass


class TextNode(PromptNode):

    def __init__(self, text:str):
        super().__init__()
        self.text = text

    def get_text(self):
        return self.text
    
    def _get_plan_step(self):
        quoted_text = self.text.replace('\\','\\\\').replace('"', '\'')
        return '{"Fixed": {"text": "' + quoted_text + '"}}'


def append(prompt_code:PromptNode, text:str) -> PromptNode:
    node = TextNode(text)
    prompt_code.add_child(node)
    return node


class BeginBlockNode(PromptNode):

    def __init__(self, id:str=None, tags=None):
        super().__init__()
        self.id = id
        self.tags = tags

    def set_end_block(self, end_block):
        self._end_block = end_block


def begin(prompt_code:PromptNode, id:str=None, tags=None) -> PromptNode:
    node = BeginBlockNode(id)
    prompt_code.add_child(node)
    return node


class BeginChatBlockNode(BeginBlockNode):

    def __init__(self, role:str, id:str=None, tags=None):
        super().__init__(id, tags)
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

def begin_chat(prompt_code:PromptNode, role:str, id:str=None, tags=None) -> PromptNode:
    node = BeginChatBlockNode(role, id=id, tags=tags)
    prompt_code.add_child(node)
    return node


class EndBlockNode(PromptNode):

    def __init__(self):
        super().__init__()
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
