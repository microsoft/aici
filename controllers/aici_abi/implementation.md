# Implementation notes

## LR(1) parsing

The LR(1) parsing consists of DFA-based lexer and the actual LR(1) parser.
DFA has a single number as the state, while the state of the LR(1) is a stack of numbers.
The LR(1) action is determined based on the next token from the lexer and the top of the stack.

The `Recognizer` interface also has a concept of stack, however every entry on that
stack contains a DFA state and an LR(1) stack.

Most of the time (~98.5% for the C grammar), pushing a byte involves only updating the DFA state,
while the LR(1) stack is copied unchanged (the memory is shared).


### Early error detection

Consider the following invalid C program:

```c
int 123456;
```

The lexer would produce `int` keyword, whitespace, `123456` constant and `;` keyword.
The parser would reject `123456`, however only after all six characters of it have been read.
This is too late for the LLM.

To detect such errors early, we compute a set of reachable tokens for each DFA state.
For example, consider a DFA that recognizes `int`, `if`, `ID` (`/[a-z][a-z0-9]*/`) and `INTLIT` (`/[0-9]+/`).
The initial DFA state has a full set of tokens, while a state after `'i'` 
has only `int`, `if`, and `ID`,
and a state after `'1'` includes only `INTLIT`.
In the picture below, each state is labelled by its reachable set,
and the token for which it is a match (if any) is postfixed with `*`. We only use lower-case letters and digits for simplicity.

```mermaid
graph LR
 0["{int,if,ID,INTLIT}"] -- "[i]" --> i(("{int,if,ID*}"))
 0 -- "[a-z] - [i]" --> id(("{ID*}"))
 0 -- "[0-9]" --> const(("{INTLIT*}"))
 const -- "[0-9]" --> const
 const -- "[a-z]" --> bot["{}"]
 i -- "[a-z0-9] - [nf]" --> id
 id -- "[a-z0-9]" --> id
 i -- "[n]" --> in(("{int,ID*}"))
 in -- "[t]" --> int(("{int*,ID}"))
 in -- "[a-z0-9] - [t]" --> id
 int -- "[a-z0-9]" --> id
 i -- "[f]" --> if(("{if*,ID}"))
 if -- "[a-z0-9]" --> id
```

For each LR(1) automaton state we compute a set of viable tokens, i.e., ones that do
not immediately lead to an error.

While parsing input, if the intersection of viable and reachable tokens is empty, we report an error.

In the example above, the viable tokens after `int` do not include `INTLIT`,
and thus the parser fails immediately at `1`.

