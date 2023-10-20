from .llm import LLM, ChatLLM
import openai


class OpenAILLM(LLM):
    def __init__(self, model_name, organization, api_key):
        super().__init__()
        self.model_name = model_name
        openai.api_key = api_key
        openai.organization = organization

    def __call__(self, prompt, constraints=None, **kwargs):
        assert constraints is None or len(constraints) == 0, "OpenAILLM does not support constraints"

        # copy kwargs and rename 'max_new_tokens' to 'max_tokens'
        kwargs = {**kwargs}
        if 'max_new_tokens' in kwargs:
            kwargs['max_tokens'] = kwargs['max_new_tokens']
            del kwargs['max_new_tokens']

        response = openai.Completion.create(
            model = self.model_name,
            prompt = prompt,
            **kwargs
        )
        return response["choices"][0]["text"]
    
    def get_name(self):
        return self.model_name



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
    
    def get_name(self):
        return self.model_name

