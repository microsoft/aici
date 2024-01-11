# aici_abi

This crate specifies the application binary inferface (ABI) for the AICI Controllers.
It also provides higher-level interfaces for implementing controllers.

## Low-level interface

Conceptually, the lowest level interface to AICI constraint is this:

```rust
type TokenId = u32;
type SeqId = u32;

trait AiciVm {
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
See the [AiciVm Rust trait](aici_abi/src/lib.rs) for details.

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

## Byte stack interface

The constraints are typically expressed on strings or bytes, not tokens.
To compute the set of tokens that match a string constraint, one needs go through all the possible tokens
and apply the constraint.
An efficient way to do this is walk a prefix tree (trie) of all tokens.
The `aici_abi` library implements this trie and exposes a way of filtering when provided with a constraints
implementing the [following interface](aici_abi/src/toktree.rs):

```rust
pub trait Recognizer {
    /// If `stack.top()` transitions via `byte` to `X`, execute `stack.push(X)`.
    fn push_byte(&mut self, byte: u8);
    /// for _ in 0..num { stack.pop() }
    fn pop_bytes(&mut self, num: usize);
    /// X = stack.top(); stack.empty(); stack.push(X)
    fn collapse(&mut self);
    /// check if stack.top() transitions via byte to a viable state
    fn byte_allowed(&mut self, byte: u8) -> bool;
    /// check if stack.top() transitions via tok to a viable state
    fn special_allowed(&mut self, tok: SpecialToken) -> bool;
    /// Called when iteration over the trie is finished
    /// Stack has exactly one element then.
    fn trie_finished(&mut self);
    /// This combines `push_byte` and `byte_allowed` into one function for performance.
    fn try_push_byte(&mut self, byte: u8) -> bool;
}
```

The `AiciRecognizer` struct converts `Recognizer` to `AiciVm`.

## Functional byte interface

The following interface can be transformed into `Recognizer` using `StackRecognizer` struct.

```rust
pub trait FunctionalRecognizer<S: Copy> {
    /// Initial state
    fn initial(&self) -> S;
    /// Extend the recognizer with given byte.
    fn append(&self, state: S, byte: u8) -> S;
    /// Check if given byte is allowed in given state.
    fn byte_allowed(&self, state: S, byte: u8) -> bool;
    /// Check if given special token is allowed in given state.
    fn special_allowed(&self, state: S, tok: SpecialToken) -> bool;
}
```

These three layers add up to about 40k of compiled code (Wasm).

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
