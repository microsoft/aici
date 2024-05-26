# Derivative based regex matcher

For basic introduction see
[Regular-expression derivatives reexamined](https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf).

For extensions, see
[Derivative Based Nonbacktracking Real-World Regex Matching with Backtracking Semantics](https://www.microsoft.com/en-us/research/uploads/prod/2023/04/pldi23main-p249-final.pdf)
and
[Derivative Based Extended Regular Expression Matching Supporting Intersection, Complement and Lookarounds](https://arxiv.org/pdf/2309.14401)
and the [sbre](https://github.com/ieviev/sbre/) implementation of it.

## Performance

We do not use rustc-hash, but the standard HashMap to limit possibility of DoS.

## TODO

- [ ] look-aheads, locations
- [ ] simplification of Byte/ByteSet in And/Or
- [ ] place limit on number of states
- [ ] alphabet size == 0 => invalid state loop

### Future improvements

- [ ] build trie from alternative over many strings in re parser
- [ ] use hashbrown raw table for VecHashMap
- [ ] more simplification rules from sbre
- [ ] tests
- [ ] benchmarks
