import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams
import pyaici

def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)

    # build it first, so it fails fast
    aici = pyaici.AiciRunner(rtpath=args.aici_rt, tokenizer=args.aici_tokenizer)

    engine = LLMEngine.from_engine_args(engine_args)
    pyaici.install_in_vllm(aici)

    # Test the following prompts.
    test_prompts: List[Tuple[str, SamplingParams]] = [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2,
                        best_of=5,
                        temperature=0.8,
                        top_p=0.95,
                        frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True,
                        temperature=0.0)),
    ]
    
    test_prompts = [
        ("Here is an example JSON about Joe Random Hacker in Seattle:\n",
         SamplingParams(temperature=0.9, n=1, max_tokens=120))
    ]
    for (prompt, params) in test_prompts:
        params.aici_module = args.aici_module
        params.aici_arg = ""

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    step_no = 0
    while True:
        step_no += 1
        # To test continuous batching, we add one request at each step.
        if test_prompts and step_no % 3 == 1:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print("")
                print("[Prompt] " + request_output.prompt)
                for out in request_output.outputs:
                    print("[Completion] " + out.text)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser.add_argument(
            '--aici-module',
            type=str,
            default='',
            help='module id')
    parser.add_argument(
            '--aici-rt',
            type=str,
            required=True,
            help='module id')
    parser.add_argument(
            '--aici-tokenizer',
            type=str,
            default='llama',
            help='tokenizer to use; llama, gpt4, ...')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
