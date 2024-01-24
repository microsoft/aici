# What is a Transparency Note?

An AI system includes not only the technology, but also the people who will use it, the people who will be affected by it, and the environment in which it is deployed. Creating a system that is fit for its intended purpose requires an understanding of how the technology works, what its capabilities and limitations are, and how to achieve the best performance. Microsoft’s Transparency Notes are intended to help you understand how our AI technology works, the choices system owners can make that influence system performance and behavior, and the importance of thinking about the whole system, including the technology, the people, and the environment. You can use Transparency Notes when developing or deploying your own system, or share them with the people who will use or be affected by your system. 

Microsoft’s Transparency Notes are part of a broader effort at Microsoft to put our AI Principles into practice. To find out more, see the [Microsoft AI principles](https://www.microsoft.com/ai/responsible-ai).


# The basics of the AI Controller Interface

## Introduction

The AI Controller Interface (AICI) is a system that improves the controllability of large language models (LLMs). With the capabilities provided by AICI, AI developers can implement a wide variety of strategies for enforcing soft and hard constraints to improve reliability of output and enforce structural constraints on generated text.

The AI Controller Interface allows special purpose controllers to tightly integrate into language model decoding (i.e., the generation of text).  This interface allows controllers to constrain decoding so that it matches a specific structure. For example, a controller might ensure that generated text matches a specified regular expression, context free grammar, or other programmatic rules.  The interface also allows controllers to direct generation by dynamically editing prompts and generated text, generating text in parallel, and backtrack/fast-forward through the text generation process.

## Key terms

**AI Controller** (or just the Controller): The controller is a program that can control language model decoding through constraints and dynamic editing of the prompts and generated text.

**AI Controller Interface:** a system that enables controllers to run in tight integration with LLM decoding inside an LLM serving engine.

**Batched or parallel inference:** This is when multiple text generations are running in parallel. Batched or parallel inference can be used to generate independent texts for separate requests, or to generate, for example, a list of items for one request.

**Constrained Decoding:** Language model decoding can be constrained to follow specific rules.  For example, if we want to generate text in all lower-case, we can modify the LLM decoding process so that only lower-case text is considered.  I.e., we can ignore probabilities for all tokens that do not match a constraint and choose the next token using the computed probabilities only for those tokens that do match the constraint.  Constraints can be expressed using regular expressions, context-free grammars, or any other programmatic rules.

**Large Language Model (LLM):** A large language model is an AI model that, given the beginning of a document (a prompt), can generate the text that is most likely to complete the document.

**LLM decoding:** LLM decoding is the process of generating text, token-by-token, using a large language model.  When decoding the next token in a sequence, the large language model generates a probability estimate for all possible next tokens.  I.e., what is the likelihood that ‘A’ is the next token, that ‘a’ is the next token, ‘B’ is the next token, etc. The final step in decoding is choosing the next token in proportion to the computed probability estimates.

**LLM serving engine:** An LLM serving engine is a (cloud) service that hosts a large language model and provides a REST API for text completion.

# Capabilities

## System behavior

The AI Controller Interface is efficient, secure, and designed for integration into cloud-hosted LLM serving engines.  Controllers are lightweight WebAssembly (Wasm) modules which run on the same machine as the LLM inference engine, utilizing the CPU while the GPU is busy with token generation.  Controllers are sandboxed and cannot access the filesystem, network, or any other resources.  Wasm modules are compiled to native code and run in parallel with the LLM inference engine, inducing only a minimal overhead to the generation process. Controllers can be written in any language that can compile to Wasm (Rust, C/C++), or be interpreted inside Wasm (Python, JavaScript).

The purpose of the AI controller interface is to extend the capabilities of LLM serving engines by providing greater control over the LLM decoding process.  AI controllers can extend LLM capabilities in generic ways, or in application-specific ways.  This extended capability can be used by higher-levels of the copilot stack (e.g., by orchestration engines, agent frameworks, and applications).  

New AI controllers may be installed by platform owners, who may wish to provide a richer interface for text completion; installed by tenants, who may have specialized needs; or even be provided by end-users per-request for specialized, efficient functionality.  Of course, permission to install new controllers can be restricted as well.

## Use cases

### Intended uses

The AI controller interface can be used in multiple scenarios. The system’s intended uses include:

-	**Generating responses that conform to a specific syntax:** For example, ensuring that responses are in a specific format, or choosing from a set of multiple choice options.
-	**Generating code that conforms to correct language syntax and semantics:** For example, ensuring that generated code passes static analysis tests.
-	**Structured reasoning:** For example, providing optimized implementation of chain-of-thought, tree-of-thought or other structured reasoning strategies.  Reasoning strategies may also selectively hide or highlight information.
-	**Forced grounding:** ensuring the generated text is grounded in text included in a prompt, or in a database included inside the controller.
-	**Multi-round or multi-agent interactions:** Multiple rounds of agent interactions can be implemented within a single call to an AI controller.

### Considerations when applying capabilities to specific higher-level tasks

When applying these uses within a specific setting, developers should take care (and test/validate) that the constraints imposed by the AI controller align with the prompt instructions and the deployed language model.  

We encourage developers of AI controllers, tenants, and end users to:

1. Perform end-to-end testing in a variety of realistic, specific scenarios.

2. Validate that deployed models, prompts, and AI controllers are jointly aligned for selected scenarios.

Furthermore, for developers creating general-purpose AI controllers, we encourage communicating the intended use cases, testing procedures, and known limits to downstream end users.

The following uses cases are not supported:

* Creating AI controllers that circumvent other platform-level AI safety constraints: for example constraining a language model to generate toxic content.

# Limitations

While AI controllers can provide simplified, task-specific APIs, can make certain LLM interactions more efficient, and can constrain LLM decoding to conform to a wide variety of constraints, controllers do not give LLMs fundamentally new capabilities.  In particular, high-level correctness properties that cannot be expressed programmatically within an AI controller may not be implementable.

## Technical limitations, operational factors and ranges

AI controllers are intended to operate within strict performance bounds for greatest efficiency.  Ideally, AI controller computation on a CPU is completed well within the time it takes for a model inference to be executed on a GPU.  If controller computation takes longer, then this will negate potential performance benefits of the AI controller interface.

# System performance and errors

While generated text may conform to the literal interpretation of a constraint, it may still fail to achieve the higher-level application-specific task. For example, a controller that requires a large language model to choose from a set of multiple choice options would be satisfied by a large language model that always chooses “C,” regardless of correctness.  And, a controller that requires an LLM generate valid code may be satisfied by English text formatted to superficially look like an algebraic expression.

## Best practices for improving system performance 

In addition to imposing constraints using an AI controller, best practices include providing clear instructions to the model using a system prompt or user prompt to encourage model outputs to conform with the constraints enforced by the AI controller. In addition, ensuring that the deployed LLM has the underlying capability is important.  For example, if an AI controller is constraining text generation to conform to python code semantics, it is important to ensure the deployed LLM has been tested and demonstrated the ability to generate python code. 

# Evaluating and integrating the AI Controller Interface for your use

After integrating the AI Controller Interface and its functionalities in an end-to-end application scenario, we recommend developers perform end-to-end responsible AI benchmarking for reliability, safety, privacy, security, fairness, and other application-specific tests to validate that controllability of the LLM is providing the desired safeguards in the developer’s targeted realistic/real-world settings.

# Learn more about responsible AI

[Microsoft AI principles](https://www.microsoft.com/en-us/ai/responsible-ai)

[Microsoft responsible AI resources](https://www.microsoft.com/en-us/ai/responsible-ai-resources)

[Microsoft Azure Learning courses on responsible AI](https://docs.microsoft.com/en-us/learn/paths/responsible-ai-business-principles/)

# Learn more about the AI Controller Interface

Read more about the AI Controller Interface at the [AICI GitHub site](https://github.com/microsoft/aici/) 

# Contact us

Give us feedback on this document by filing an [issue](https://github.com/microsoft/aici/issues)

# About this document

© 2024 Microsoft Corporation. All rights reserved. This document is provided "as-is" and for informational purposes only. Information and views expressed in this document, including URL and other Internet Web site references, may change without notice. You bear the risk of using it. Some examples are for illustration only and are fictitious. No real association is intended or inferred.

This document is not intended to be, and should not be construed as providing. legal advice. The jurisdiction in which you’re operating may have various regulatory or legal requirements that apply to your AI system. Consult a legal specialist if you are uncertain about laws or regulations that might apply to your system, especially if you think those might impact these recommendations. Be aware that not all of these recommendations and resources will be appropriate for every scenario, and conversely, these recommendations and resources may be insufficient for some scenarios.

Published: 1/23/2024

Last updated: 1/23/2024
