# run as:
# PYTHONPATH=.. python genarg.py 
import ujson
import pyaici._ast as ast

def main():
    aa = {
        "steps": [ast.fixed("Here's some JSON about J.R.Hacker from Seattle:\n")]
        + ast.json_to_steps(
            {
                "name": "",
                "valid": True,
                "description": "",
                "type": "foo|bar|baz|something|else",
                "address": {"street": "", "city": "", "state": "[A-Z][A-Z]"},
                "age": 1,
                "fraction": 1.5,
            }
        )
    }

    aa = {
        "steps": [
            ast.fixed("The word 'hello'"),
            ast.label("lang", ast.fixed(" in French is translated as")),
            ast.gen(rx=r" '[^']*'", max_tokens=15, set_var="french"),
            ast.fixed(" or", following="lang"),
            ast.gen(rx=r" '[^']*'", max_tokens=15, set_var="blah"),
            ast.fixed("\nResults: {{french}} {{blah}}", expand_vars=True),
        ]
    }
    
    ast.clear_none(aa)
    print(ujson.dumps(aa))

main()