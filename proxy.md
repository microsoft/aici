# Client-side access to AICI

The [Artificial Intelligence Controller Interface (AICI)](https://github.com/microsoft/aici)
can be used to constrain output of an LLM in real time.
While the GPU is working on the next token of the output, the AICI VM can use the CPU to
compute a user-provided constraint on the next token.
This adds minimal latency to the LLM generation.

TODO: add more info
