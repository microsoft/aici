# Low-level Guidance (llguidance)

This controller implements a context-free grammar parser with Earley's algorithm
on top of a lexer which uses [derivatives of regular expressions](../derivre/README.md).

It's to be used by next-generation [Guidance](https://github.com/guidance-ai/guidance) grammars.
See how it works in [plan.md](./plan.md).

Guidance branch: https://github.com/hudson-ai/guidance/tree/lazy_grammars

## Guidance implementation notes

- `gen()` now generates a new node, `Gen`
- grammar is serialized to JSON, see `ll_serialize()`

## Status in Guidance

- [x] `save_stop_text=` doesn't work on `gen()`
- [ ] `substring()` needs to be re-implemented (translate to RegexAst)
- [ ] translate `commit_point(grammar)` into a RegexAst if
      (the grammar is non-recursive;
      no actually hidden stop strings;
      and no captures)?

## TODO

- [ ] `to_regex_vec()` in lexerspec.rs - non-contextual keywords
- [x] fix derivative computation to be non-recursive (critical for `substring()`)
- [x] add stats about how many parser transitions are made in a token trie traversal

## Only valid tokens

See https://github.com/hudson-ai/guidance/issues/13

- [ ] implement `.forced_byte()` method in `derivre`
- [ ] use this for cheap `.forced_byte()` impl in `llguidance`
- [ ] while walking token trie, remember all forced paths (there shouldn't be too many of them)

In toktrie walk, if we encounter a forced byte, we go into forced mode
where we just chase all forced bytes.
The first token we find on this path we put on some list.
We do not add any of these tokens to the allow set.

Then, after token trie walk, for every token on this list we re-create
the forced byte string, tokenize, chop excessive tokens, and add the first
token from tokenization to allow set and remaining tokens (if any) as conditional
splice.
