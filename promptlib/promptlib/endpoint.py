
from .prompt import PromptNode
from .endpoints import Endpoint

class EndpointNode(PromptNode):

    def __init__(self, endpoint:Endpoint): 
        super().__init__()
        self.endpoint = endpoint

    def runAll(self):
        plan = self.build_tree_plan()
        return self.endpoint.run(plan) # TODO we should parse through the results and integrate them back into the state of the PromptNode object

    def run(self, promptNode:PromptNode):
        plan = promptNode.build_linear_plan()
        return self.endpoint.run(plan) # TODO we should parse through the results and integrate them back into the state of the PromptNode object
    
def set_endpoint(prompt_code:PromptNode, ep:Endpoint) -> PromptNode:
    epNode = EndpointNode(ep)
    prompt_code.add_child(epNode)
    return epNode