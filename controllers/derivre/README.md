# Derivative based regex matcher

For basic introduction see
[Regular-expression derivatives reexamined](https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf).

For extensions, see
[Derivative Based Nonbacktracking Real-World Regex Matching with Backtracking Semantics](https://www.microsoft.com/en-us/research/uploads/prod/2023/04/pldi23main-p249-final.pdf)
and
[Derivative Based Extended Regular Expression Matching Supporting Intersection, Complement and Lookarounds](https://arxiv.org/pdf/2309.14401)
and the [sbre](https://github.com/ieviev/sbre/) implementation of it.

## Usage

This library uses [regex-syntax](https://docs.rs/regex-syntax/latest/regex_syntax/)
for regular expression parsing.
This means that currently there is no surface syntax for `&` and `~` operators
(but the library supports it).

The library only checks if the regex matches the string from the beginning
(it doesn't search for it, or in other words there's an implied `\A` at the beginning).

```rust
let mut rx = RegexVec::new_single("[ab]c").unwrap();
assert!(rx.is_match("ac"));
assert!(rx.is_match("bc"));
assert!(!rx.is_match("xxac"));
assert!(!rx.is_match("acxx"));
```

The library supports a single look-ahead at the end of the regex,
written as `A(?P<stop>B)` where `A` and `B` are regexes without any look-arounds.
You can get the length of the string matching `B` upon successful match.

```rust
// the syntax in other libraries would be: r"\A[abx]*(?=[xq]*y)"
let mut rx = RegexVec::new_single("[abx]*(?P<stop>[xq]*y)").unwrap();
assert!(rx.lookahead_len("axxxxxy") == Some(1));
assert!(rx.lookahead_len("axxxxxqqqy") == Some(4));
assert!(rx.lookahead_len("axxxxxqqq") == None);
assert!(rx.lookahead_len("ccqy") == None);
```

You can also run several regexes at once, [see sample](./tests/sample_multi.rs).

## TODO

- [x] look-aheads, locations
- [x] simplification of Byte/ByteSet in Or
- [ ] place limit on number of states
- [ ] alphabet size == 0 => invalid state loop

### Future improvements

- [ ] build trie from alternative over many strings in re parser
- [x] use hashbrown raw table for VecHashMap
- [ ] more simplification rules from sbre
- [ ] tests
- [ ] benchmarks
