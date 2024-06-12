# Accelerated Guidance Grammars (AG2)

This controller implements a context-free grammar parser with Earley's algorithm
on top of a lexer which uses [derivatives of regular expressions](../derivre/README.md).

It's to be used by next-generation [Guidance](https://github.com/guidance-ai/guidance) grammars.
See how it works in [plan.md](./plan.md).

Guidance branch: https://github.com/paulbkoch/guidance/tree/lazy_grammars

## Guidance implementation notes

- `gen()` now generates a new node, `Gen`
- grammar is serialized to JSON, see `ag2_serialize()`

## Status in Guidance

- [ ] `gen_mode` in `_gen.py` needs to become a flag on the model/engine
- [ ] `gen_json()` needs to be re-implemented
- [ ] `save_stop_text=` doesn't work on `gen()`
- [ ] `substring()` needs to be re-implemented (translate to RegexAst)
- [ ] translate `commit_point(grammar)` into a RegexAst if
      (the grammar is non-recursive; 
      no actually hidden stop strings; 
      and no captures)?

## TODO

- [ ] `to_regex_vec()` in lexerspec.rs - non-contextual keywords
- [x] handle stop tokens in `gen_grammar()` - stop tokens removed
- [ ] use `RegexAst::Byte(0xff)` for `EOS_MARKER`
- [ ] fix derivative computation to be non-recursive (critical for `substring()`)
- [ ] add stats about how many parser transitions are made in a token trie traversal
