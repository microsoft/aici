import pyaici.server as aici

# note that VSCode syntax highlighting is not perfect r""" ... """ (it assumes regexp)
c_yacc = r"""

%start T
%%

SKIP
    : "/\n?[ \t\v\f]*/" ; // white-space, newline, tabs, ...

field
    : "/[a-zA-Z_][a-zA-Z0-9_]*/" ;

typeConst
    : "/[A-Z_][A-Z0-9_]*/" ;

typeVar
    : "/[a-z]/" ;

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
    // operators
    | "(" List ")" "=>" T
    // constant types (uninterpreted types)
    | typeConst
    // type variables
    | typeVar
    // parentheses, e.g., to change associativity of functions
    | "(" T ")"
    ;

%%
"""

sample_apalache = """
(Str, Bool) => Bool
Set(Int) -> Set(Int)
"""

async def test_grammar():
    await aici.FixedTokens("Start")


tokens = aici.tokenize(sample_apalache)
print(tokens)
constr = aici.CfgConstraint(c_yacc)
outp = []
for t in tokens:
    if not constr.token_allowed(t):
        print(f"Token {t} not allowed")
        print("OK: " + repr(aici.detokenize(outp).decode("utf-8", "ignore")))
        print("fail: " + repr(aici.detokenize([t]).decode("utf-8", "ignore")))
        break
    outp.append(t)
    constr.append_token(t)
print(f"EOS allowed: {constr.eos_allowed()}")

aici.test(test_grammar())
print("Hello")
