from .prompt import PromptNode
from .model import ModelNode, ChatModelNode
from .constraints import Constraint

class ConstraintNode(PromptNode):
    def __init__(self, constraint:Constraint):
        super().__init__()
        self.constraint = constraint

    def set_parent(self, parent): 
        super().set_parent(parent)
        model = self.get_parent_of_type(ModelNode)

        # TODO this will fail in cases where we are building subprompts and then only later prepending a model
        assert model is not None, "ConstraintNode must have an ancestor of type ModelNode"
        assert model.llm.supports_constraints, "Model must support constraints"
        self.is_chat = isinstance(model, ChatModelNode)

    def _get_plan_step(self):
        dict = {"type": self.constraint.get_type()}
        dict.update(self.constraint.get_constraint_args())
        dict.update(self._get_attributes())
        return {"constraint": dict}



def constrain(prompt_code:PromptNode, constraint:Constraint) -> PromptNode:
    node = ConstraintNode(constraint)
    prompt_code.add_child(node)
    return node
