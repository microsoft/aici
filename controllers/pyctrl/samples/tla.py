import pyaici.server as aici
 
apalache = r"""
 
%start tnl
%%

List
    : T
    | List ", " T
    ;

tnl: T "/\n/";

T
    // integers
    : "Int"
    // immutable constant strings
    | "Str"
    // boolean
    | "Bool"
    // functions
    | T " -> " T
    // sets
    | "Set" "(" T ")"
    // sequences
    | "Seq" "(" T ")"
    // tuples
    | "<<" List ">>"
    // parentheses, e.g., to change associativity of functions
    | "(" T ")"
    // operators
    | "(" List ") => " T
    ;
 
%%
"""
 
 
async def gen_and_test_grammar():
    aici.log_level = 3
    await aici.FixedTokens(
        """
Here's a TLA+ spec:

---- MODULE Counter ----
VARIABLE
        b
        q

Init == b = TRUE
Next == q = q + 1
====

Now with added types:

---- MODULE Counter ----
VARIABLE
        b: """
    )
    await aici.gen_tokens(yacc=apalache, max_tokens=10, store_var="b")
    await aici.FixedTokens("        q: ")
    await aici.gen_tokens(yacc=apalache, max_tokens=10, store_var="q")

 
aici.test(gen_and_test_grammar())