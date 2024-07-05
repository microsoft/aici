# Plan for fast server-side guidance

The main idea is introducing lexer in addition to the existing Earley parser.

It's implemented in two steps:

- we alter the semantics of `gen()` without `stop=` to produce what we call _lazy grammars_
- we introduce _greedy grammars_,
  which allow for defining programming language grammars with standard lexer rules;
  these are defined inside `greedy_grammar()` functions

The `gen()` with `stop=` (or equivalent `stop_regex=`, `suffix=`, etc.) behaves exactly as before.
In Guidance README.md, the only uses of `gen()` without `stop=` are at the end of the grammar,
where the `stop=EOS` is implied.
Thus, lazy grammars cover all of the README.md except for the CFG example
(covered better by the greedy grammars) and tools (which are not stateless).

## Lazy grammars

The Earley parser will now use _lexemes_ instead of bytes as terminals.
(Lexemes are often called tokens, but we avoid that term since it means something else for LLMs).
Lexemes are defined by regular expressions
and together define a lexer (often also called scanner or tokenizer).

The main thing about lexer is that it's unambiguous: there is only one way to turn a string into lexemes.
Here, we use a lazy lexer which takes the shortest match for every lexeme,
whereas for greedy grammars we take the longest.

To find the lexemes,
we search in the grammar for all literal strings and regexes inside of `gen()` invocations,
and treat them all as regexes.
For `gen(regex=A, stop=B)` the regex is `AB`,
and for string literal the regex is string literal appropriately quoted.
These are lexemes (terminals) in the CFG grammar.

Then we run the Earley parser as usual.
If a row allows for scanning lexemes `L1`, `L2`, ..., `Ln`,
we run the regular expression `L1|L2|...|Ln` on the input
and stop on the first match (lazy matching).
Then we run the Earley parser scan as if we found the lexeme(s) that matched.

For example:

```python
"Name: " + gen(regex=r'[A-Z][a-z]+', stop='\n')) + "\n" +
select([
  "Married? " + gen(regex=r'Yes|No'),
  "Age: " + gen(regex=r'\d+', stop='\n')
]) + "\n"
```

Lexemes (regexes):

```python
L0 = r"Name: "
L1 = r"[A-Z][a-z]+\n"
L2 = r"\n"
L3 = r"Married\? " # (note quote on '?')
L4 = r"Yes|No"
L5 = r"Age: "
L6 = r"\d+\n"
```

There is no `L7 = r"\n"`, since it's already covered by `L2`.

Grammar:

```python
L0 + L1 + L2 + select([L3 + L4, L5 + L6]) + L2
```

As CFG rules:

```
start   : L0 L1 L2 select L2
select  : L3 L4
        | L5 L6
```

When parsing, we first match `L0 = r"Name: "`,
then `L1 = r"[A-Z][a-z]+\n"`, then `L2 = r"\n"`,
since these are the only ones allowed by the Earley parser.
Then, the parser will allow `L3 = r"Married\? "` and `L5 = r"Age: "`,
so we run regex `L3|L5` (`r"Married\? |Age: "`).
Depending on which one matched (in this case it cannot be the case
that both match), we match either `L4 = r"Yes|No"` or `L6 = r"\d+\n"` (not `L4|L6`).
Finally, we match `L2 = r"\n"` again.

### Notes

- we need to require all regexes to never match an empty string
- in reality, we build one automaton for all regexes, and mess with states
  to only get the subset we need, instead of building `L1|L2|...` dynamically
- if people do things like `zero_or_more(" ")`, this will be as slow as the current
  implementation (or slower); we need to encourage them to use `gen()` with
  regexes instead of using single-character strings
- if there is no `stop`, we can infer the `stop` to be the next character that
  comes after the `gen()` (this gets rid of vast majority of `stop` in README.md,
  and all in the example above)
- we no longer need `commit_point()` for the `stop`, as they are always lazy;
  we can probably get rid of it altogether
- (maybe) as an optimization, the grammar can be partitioned at the top-level joins,
  to limit the size of lexer automatons (in this case, we would have four grammars:
  `L0`, `L1`, `L2`, and `select`); this depends on how we handle the switch
  between grammars though

### Lexeme prefix problem

If one lexeme can be a prefix of another, and both are enabled in a given
row, the shorter lexeme will win.
For example, here the `"foobar"` will never be selected:

```python
select(["foo", "foobar", "baz"])
```

It can be rewritten as:

```python
select(["foo" + select(["", "bar"]), "baz"])
```

We can likely do some of these rewrites automatically.
They also don't seem very common in practice.

See also _Lexeme conflict detection_ below.

## Greedy grammars

Greedy grammars are defined by `greedy_grammar(grammar, skip=whitespace_regex)` function.
They are invoked by `gen(grammar=...)` from a lazy grammar (TBD).
For example:

```python
def identifier():
  return regex(r'[a-zA-Z_][a-zA-Z0-9_]*')

def sql_program():
  return select([
    "SELECT" + identifier() + "FROM" + identifier() + 
    optional("WHERE" + identifier() + "=" + ...),
    ...
  ])

def sql():
  return greedy_grammar(sql_program(), skip=r'\s*')

# lazy grammar here
def my_program():
  return f"""
Query to list all animals
```sql
{gen(grammar=sql(), suffix="```")}
"""
```

The parsing is similar to the lazy grammar,
but when constructing lexeme list,
we look for `regex()` nodes (`gen()` nodes are not allowed) and strings.
The main difference is that instead of the shortest match, we take the longest when running the lexer,
(as is common with programming language lexers)
and we skip over anything that matches the `skip` regex.

From the point of view of the upper-level parser,
the `greedy_grammar()` construct is a single token,
so the lexers do not interact with each other.

The stop token, passed in from the outer `gen()` call is added to the lexer.
We need to make sure it doesn't interact with the grammar lexer,
or else fix the stop token at the grammar level.

### Always-on lexemes

To match semantics of most programming languages,
by default we would enable all lexemes in all rows
(so that "while" is always parsed as a keyword and never an identifier).
Some lexemes could be marked as "contextual", meaning only enabled
in certain position (eg., in C# "get" and "set" are such lexemes).

In lazy grammars all lexemes are contextual.

## Lexeme conflict detection

In each row, we only enable lexemes that are allowed by the Earley parser
(except for the non-contextual lexemes in greedy grammars, but let's ignore that).
To give feedback to the user, and possibly split certain lexemes
(see _Lexeme prefix problem_ above), it would be good if we could detect
if two lexemes ever occur together in a row.

Formally, the terminals `A` and `B` conflict in a context-free language `L`
if there exists two words (sequences of terminals)
`wAu` and `wBv` in `L` for some words `w`, `u`, and `v`.

Unfortunately, this problem is undecidable in general
(reduce to emptiness of intersection of two context-free languages,
or directly to the Post correspondence problem).

Thus, it would be good to have a heuristic for under-approximating this problem.
That is, given a grammar `G` and two lexemes `A` and `B`,
if the algorithm returns `OK`, then `A` and `B` do not conflict in `L(G)`,
and if the algorithm returns `FAIL`, then we do not know.
