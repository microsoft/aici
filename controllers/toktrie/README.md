# toktrie - Token utility library

This crate provides a utility library for working with tokens and token tries.

## Byte stack interface

The constraints are typically expressed on strings or bytes, not tokens.
To compute the set of tokens that match a string constraint, one needs go through all the possible tokens
and apply the constraint.
An efficient way to do this is walk a prefix tree (trie) of all tokens.
The `aici_abi` library implements this trie and exposes a way of filtering when provided with a constraints
implementing the [following interface](src/toktree.rs):

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

The `AiciRecognizer` struct converts `Recognizer` to `AiciCtrl`.

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

