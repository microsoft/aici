import sys
import pyaici.server as aici

# note that VSCode syntax highlighting is not perfect r""" ... """ (it assumes regexp)
apalache = r"""

%start T
%%

SKIP
    : "/\n?[ \t\v\f]*/" ; // white-space, newline, tabs, ...

List
    : T
    | List "," T
    ;

T
    // integers
    : "Int"
    // immutable constant strings
    | "Str"
    // boolean
    | "Bool"
    // functions
    | T "->" T
    // sets
    | "Set" "(" T ")"
    // sequences
    | "Seq" "(" T ")"
    // tuples
    | "<<" List ">>"
    // parentheses, e.g., to change associativity of functions
    | "(" T ")"
    // operators
    | "(" T ")" "=>" T
    | "(" List ")" "=>" T
    ;
%%
"""

t12 = r"""
  
// Prefix all reguluar expressions to disambiguate them.
// The tool first tries a regex and only then looks at the grammar. :-(
field
    : "/f[a-zA-Z_][a-zA-Z0-9_]*/" ;

typeConst
    : "/t[A-Z_][A-Z0-9_]*/" ;

typeVar
    : "/v[a-z]?/" ;

aliasName
    : "/a[a-z]+(?:[A-Z][a-z]*)*/" ;

T12  
    // type all rules
    : "$" aliasName
    // constant types (uninterpreted types)
    | typeConst
    // type variables
    | typeVar
    // A new record type with a fully defined structure.
    // The set of fields may be empty. If typeVar is present,
    // the record type is parameterized (typeVar must be of the 'row' kind).
    // | "{" field ":" T "," ..."," field ":" T ["," typeVar] "}"
    // A variant that contains several options,
    // optionally parameterized (typeVar must be of the "row" kind).
    // | variantOption "|" ... "|" variantOption "|" [typeVar]
    // A purely parameterized variant (typeVar must be of the "row" kind).
    | "Variant" "(" typeVar ")"
    ;
  
%%
"""


# async def gen_and_test_grammar():

async def test_grammar():
    await aici.FixedTokens("Start")

aici.test(test_grammar())


## define a list of test inputs
inputs = [
    "Int",
    "Int",
    "Str",
    "(Int)",
    "Bool",
    "Int -> Int",
    "Set(Int)",
    "Seq(Int)",
    "Set(Int) -> Set(Int)",
    "Set(<<Int, Int>> -> Int)",
    "<<Int,Int>>",
    "(Int,Int) => Int",
    "(Int,Bool) => Int",
    "((Int,Bool) => Int) => Bool",
    "$aalias",
]

## loop over the inputs and test the grammar
for input in inputs:
    print(f"Testing input: {input}")
    tokens = aici.tokenize(input)
    # print(tokens)
    constr = aici.CfgConstraint(apalache)
    outp = []
    for t in tokens:
        if not constr.token_allowed(t):
            ## Abort/terminate if token is not allowed.
            print(f"Token {t} not allowed")
            print("OK: " + repr(aici.detokenize(outp).decode("utf-8", "ignore")))
            print("fail: " + repr(aici.detokenize([t]).decode("utf-8", "ignore")))
            sys.exit(1)
        outp.append(t)
        constr.append_token(t)
    # print(f"EOS allowed: {constr.eos_allowed()}")
