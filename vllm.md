# Notes on integration with vLLM

Following are callbacks in the vLLM flow:

```python
LLMEngine.step()
    Scheduler.schedule()
        scheduler_outputs = Scheduler._schedule()
        SamplingParams.initiate_step(scheduler, llm_engine.counter, scheduler_outputs)
        return SequenceGroupMetadata(scheduler_outputs)
    samples = LLMEngine._run_workers("execute_model")
        Worker.execute_model
            Worker._prepare_inputs
                # ...
                SamplingParams.recv_attention_mask()
                # ...
            self.model()
                # ...
                Sampler.forward()
                    logits = ...
                    SamplingParams.apply_dynamic_logit_bias(logits)
                    return _sample(logits) : SequenceOutputs
    return LLMEngine._process_model_outputs(samples)
        LLMEngine._process_sequence_group_samples()
            SamplingParams.append_ff_tokens(seq_group)
            # free and fork sequences as needed
        SamplingParams.finish_sampling()
        json_output = ...
        return json_output
```

Thoughts:
- expose Scheduler._schedule() and call it from LLMEngine; move initiate_step to LLMEngine
- return logits from Sampler.forward() and call _sample() from LLMEngine; move apply_dynamic_logit_bias to LLMEngine
- pass attn_mask to execute model from LLMEngine

- vllm forks sequences in _process_sequence_group_samples(); this means fork processing in AICI is done
  in pre_process(), not process(), so it blocks; in full AICI env you would only fork from AICI module not
  n= parameter