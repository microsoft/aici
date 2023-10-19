import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from .llm import LLM

class TransformersLLM(LLM):
    def __init__(self, model_name, **model_args):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        self.supports_constraints = True

    def supportsConstraints():
        return True
    
    def __call__(self, prompt, constraints=None, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        if constraints is None or len(constraints) == 0:
            #inputs = inputs.to(device)
            output = self.model.generate(inputs["input_ids"], **kwargs)
            ret = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return ret[len(prompt):]
            #return tokenizer.batch_decode(output.tolist())
        else:
            return self.constrainedGeneration(prompt, constraints, **kwargs)


    ### EMK: We will give the suffix to the constraint provider when asking for the next token
    ###      When the constraint provider reaches the end of an option/program/etc, it will append the suffix    
    def recursive_getnext_valid_token(self, constraint, validstr=""):
        ret = []
        # TODO - this logic assumes that, if 'abc' is a valid token, that 'ab' must also be a valid token.  Not sure this is true. If not, the answer is to just use the underlying token mapping directly.
        next_valid_chars = constraint.valid_next_chars()
        if len(next_valid_chars) > 0:
            for c in next_valid_chars:
                valid_tokens = self.tokenizer.encode(str(validstr + c))
                if( len(valid_tokens) > 1):
                    #print(f"valid token: {valid_tokens[0]}")
                    ret.append(valid_tokens[0])
                else:
                    # NOTE: adding the abbreviated token valid_tokens[0] when a longer one is available doesn't seem to work correctly ? TODO why
                    next_constraint = constraint.set_next_chars(c)
                    recursive_valid_tokens = self.recursive_getnext_valid_token(next_constraint, validstr + c)
                    if( len(recursive_valid_tokens) > 0 ):
                        ret.extend(recursive_valid_tokens)
                    else: 
                        # if there are no valid tokens after this one, then our greedy search for the next token is done
                        # TODO - we should be doing this not only when recursive_valid_tokens is empty, but whenever
                        #        not generating a next token is valid.  Ie., we could continue, or we could stop...
                        # TODO - integrate this with the suffix logic TODO below 
                        ret.append(valid_tokens[0])
            return set(ret)
        else:
            # TODO pull characters from suffix.
            return []

    def gen_next_token(self, current_prefix, valid_next_tokens):
        logit_bias = {}
        for token in valid_next_tokens:
            logit_bias[token] = 100

        #print(f"Current prefix = {current_prefix}")
        # TODO --- the next line is not calling the model correctly
        # LOOK UP HOW TO DO IT CORRECTLY GIVEN A HF TRANSFORMERS MODEL
        # check out guidance.endpoints._transformers.py line 335. 
        # it's a call to self.model.generate() but with a lot of special arg setup?
        #gen_obj = self.model.generate(
        #    current_prefix, # TODO: perhaps we should allow passing of token ids directly? (this could allow us to avoid retokenizing the whole prefix many times)
        #    max_tokens=1,
        #    logit_bias=logit_bias,
        #    logprobs=len(logit_bias),
        #    cache_seed=0,
        #    stream=False,
        #    token_healing=False # we manage token boundary healing ourselves for this function
        #)

        input_ids = self.tokenizer.encode(current_prefix, return_tensors="pt")
        # TODO take care of moving input_ids to the correct device

        processors = []
        processors.append(BiasLogitsProcessor(self.model, len(self.tokenizer.vocab), logit_bias))

        generate_args = dict(
            inputs=input_ids,
            # attention_mask
            # position_ids
            max_new_tokens=1,
            top_p = 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=transformers.LogitsProcessorList(processors),
            #stopping_criteria=
            output_scores=True,
            return_dict_in_generate=True            
        )

        # temperature and do_sample

        gen_obj = self.model.generate(**generate_args)

        generated_text = self.tokenizer.decode([gen_obj.sequences[0].tolist()[-1]], skip_special_tokens=True, )

        # TODO - is setting logit_biases sufficient? Or do we still need to go through the gen_obj and ensure that returned tokens meet our constraints?

        return generated_text
    
    def constrainedGeneration(self, prompt, constraints, **kwargs):
        assert len(constraints) == 1, "TransformersLLM only supports a single constraint"

        gen = prompt
        current_constraint = constraints[0]

        # extract max_tokens from kwargs, otherwise default to 100
        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 100))

        # TODO extend this to do a beam search, or if we have a small cheap model, we could use it for an A* search?
        for i in range(max_tokens):
            next_valid_tokens = self.recursive_getnext_valid_token(current_constraint)
            if( len(next_valid_tokens) == 0):
                # the only way to get here is if the constraint provider has reached the end of an option/program/etc
                # TODO add ways for the LLM to end early, generate the suffix, etc.
                break

            next_gen = self.gen_next_token(gen, next_valid_tokens) 
            next_token = next_gen # next_gen["text"]
            next_chars = next_token # the token is returned as characters, not as a token id
            gen += next_chars
            current_constraint = current_constraint.set_next_chars(next_chars)

        gen = gen[len(prompt):]
        return gen
        #lm[name] = gen
        #
        ## see if we are appending to a list or not
        #if list_append:
        #    value_list = lm.get(name, [])
        #    value_list.append(gen)
        #    lm[name] =  value_list
        #else:
        #    lm[name] = gen



class BiasLogitsProcessor():
    """ Simple token biasing.
    """

    def __init__(self, model, vocab_size, logit_bias):
        """ Build a new BiasLogitsProcessor.
        """
        import torch
        
        self.bias_vector = torch.zeros(vocab_size)
        for token, bias in logit_bias.items():
            self.bias_vector[token] = bias
        self.bias_vector = self.bias_vector.to(model.device)

    def __call__(self, input_ids, scores):

        # handle list inputs
        if isinstance(scores, list):
            import torch
            scores = torch.tensor(scores)

        return scores + self.bias_vector