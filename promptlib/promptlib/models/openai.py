from .llm import ChatLLM
import openai

class OpenAIChatLLM(ChatLLM):
    def __init__(self, model_name, organization, api_key):
        super().__init__()
        self.model_name = model_name
        openai.api_key = api_key
        openai.organization = organization

    def __call__(self, messages, constraints=None, **kwargs):
        assert constraints is None or len(constraints) == 0, "OpenAIChatLLM does not support constraints"

        response = openai.ChatCompletion.create(
            model = self.model_name,
            messages = messages,
            **kwargs
        )
        return response["choices"][0]["message"]["content"]

