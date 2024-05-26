# Derivative based regex matcher

For basic introduction see
[Regular-expression derivatives reexamined](https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf).

For extensions, see
[Derivative Based Nonbacktracking Real-World Regex Matching with Backtracking Semantics](https://www.microsoft.com/en-us/research/uploads/prod/2023/04/pldi23main-p249-final.pdf)
and
[Derivative Based Extended Regular Expression Matching Supporting Intersection, Complement and Lookarounds](https://arxiv.org/pdf/2309.14401)
and the [sbre](https://github.com/ieviev/sbre/) implementation of it.

## TODO

- [ ] look-aheads, locations
- [ ] use regex-syntax crate to create Expr
- [ ] the actual matching of strings
- [ ] add matcher on a vector of Expr (lexer style)  
- [ ] simplification of Byte/ByteSet in And/Or
- [ ] more simplification rules from sbre
- [ ] tests
- [ ] benchmarks
