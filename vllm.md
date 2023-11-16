# Notes on integration with vLLM

Following are callbacks in the vLLM flow:

```python
LLMEngine.step()
    Scheduler.schedule()
        scheduler_outputs = Scheduler._schedule()
        SamplingParams.initiate_step(scheduler, llmengine.counter, scheduler_outputs)
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
        json_output = ...
        return json_output
```
