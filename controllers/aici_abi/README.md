# aici_abi

This crate specifies the application binary interface (ABI) for the AICI Controllers.
It also provides higher-level interfaces for implementing controllers.

## Low-level interface

Conceptually, the lowest level interface to AICI constraint is this:

```rust
type TokenId = u32;
type SeqId = u32;

trait AiciCtrl {
    /// Called with the initial prompt. ~1000ms time limit.
    fn init_prompt(prompt: Vec<TokenId>);

    /// Called before mid_process(), can fork or suspend. ~1ms.
    fn pre_process() -> enum {
        Stop,
        Continue, // Same as Fork { num_forks: 1 }
        Suspend,  // skip this generation round
        Fork { num_forks: u32 },
    }

    /// This is the main entry point for the module. ~20ms.
    fn mid_process(fork_group: Vec<SeqId>) -> enum {
        Stop,
        SampleWithBias { bias: Vec<f32> },
        Splice { backtrack: u32, ff_tokens: Vec<TokenId> }
    };

    /// Called after tokens are appended. ~1ms.
    fn post_process(tokens: Vec<TokenId>) -> enum { Stop, Continue };
}
```

Tokens depend on the tokenizer used (eg., for Llama there 32000 tokens, and for GPT-4 there is ~100k).

The actual binary interface is a bit more complicated, due
to limitations in passing values to and from Wasm.
A Wasm module instance is created for each token sequence.
Also, when the sequence forks (as in beam search), the module instance is cloned.
See the [AiciCtrl Rust trait](src/lib.rs) for details.

A number of functions are exposed to the Wasm module.

First, there are functions for accessing the current tokenizer:

```rust
/// Given a byte sequence, return a sequence of token Ids.
fn tokenize_bytes(s: Vec<u8>) -> Vec<TokenId>;

/// Represents trie of all tokens in the current tokenizer.
impl TokTrie {
    /// Get Id for EOS token etc.
    fn special_token(tok: SpecialToken) -> TokenId;
    /// Number of tokens.
    fn vocab_size() -> usize;
    /// Convert token Id to bytes (often UTF-8 string).
    fn token(token: TokenId) -> Vec<u8>;
    /// Given a Recognizer, compute the set of allowed tokens.
    fn compute_bias(rec: impl Recognizer) -> Vec<bool>;
}
```

Different forks in a sequence can communicate via shared variables:

```rust
/// This can be looked up in fork_group.
fn self_seq_id() -> SeqId;

trait VariableStorage {
    fn get(name: str) -> Option<Vec<u8>>;
    fn set(name: str, value: Vec<u8>);
    fn append(name: str, value: Vec<u8>);
}
```

Additionally, the `stdout` and `stderr` file descriptors are captured by the runtime
and returned to user when streaming results.

This interface may need to be extended in the future.

See the `toktrie` crate for general utilities for building constraints.
This crate implements a few constraints including regexes, LR(1) grammars, and
substrings.


## Regular expressions

The `FunctionalRecognizer` interface is implemented for regular expressions.
The `S` type is the state of the DFA (Deterministic Finite Automaton) that recognizes the regular expression,
then `append()` and `byte_allowed()` are the standard DFA operations,
while `special_allowed()` is only implemented for end-of-sequence token
(which is allowed when the current state is accepting).

## LR(1) grammars

The `Recognizer` interface is implemented for LR(1) grammars and DFA-based lexers.

The grammar uses inline syntax for the lexer:

- `"keyword"` or `'keyword'` for keywords; any string works, eg. `"+="`, `"while"`, ...
- `"/.../"` or `'/.../'` for regular expressions; you cannot have both `'` and `"` in the regex
  Special `SKIP` rule is used to indicate tokens that need to be skipped by the LR(1) parser (eg., whitespace and comments)

The lexer has a DFA which recognizes all regexps and keywords
(a big disjunction, but with additional machinery to disambiguate between different branches).
It goes byte by byte, until the DFA gets to a dead state (from which no match is possible).
Then it goes back one byte and checks for match.
It prefers keywords over regexps.
If no match is found, an error is reported, which requires careful design of the lexical part of the grammar
(eg., see how the `white-space` rule below is prefix of the `pre-processor` rule).

For example, this is fragment of [grammar for C](./grammars/c.y):

```yacc
%start translation_unit
%%

SKIP
    : "//\*[^*]*\*+([^/*][^*]*\*+)*//"  // block comment
    | "///.*/"                          // line comment
    | "/\n[ \t\v\f]*#(.*\\\n)*.*/"      // pre-processor
    | "/\n?[ \t\v\f]*/"                 // white-space
    ;

IDENTIFIER: "/[a-zA-Z_][0-9a-zA-Z_]*/" ;

CONSTANT
        : "/0[xX][0-9a-fA-F]+[uUlL]*?/"
        | "/0[0-9]+[uUlL]*?/"
        ;

STRING_LITERAL: '/"(\\.|[^\\"])*"/' ;

primary_expression
    : IDENTIFIER
    | CONSTANT
    | STRING_LITERAL
    | "(" expression ")"
    ;

// ...

enum_specifier
    : "enum" "{" enumerator_list "}"
    | "enum" IDENTIFIER "{" enumerator_list "}"
    | "enum" IDENTIFIER
    ;

// ...

translation_unit
    : external_declaration
    | translation_unit external_declaration
    ;
```
