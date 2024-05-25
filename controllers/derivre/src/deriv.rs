use crate::{
    ast::{Expr, ExprRef},
    hashcons::HashCons,
};

struct Regex<'a> {
    exprs: HashCons<'a, Expr<'a>>,
}

impl<'a> Regex<'a> {
    pub fn new() -> Self {
        Regex {
            exprs: HashCons::new(),
        }
    }

    pub fn empty_string(&mut self) -> ExprRef {
        self.exprs.insert(Expr::EmptyString)
    }

    pub fn mk_and(&mut self, es: &'a [ExprRef]) -> ExprRef {
        self.exprs.insert(Expr::And(es))
    }
}

fn foo() -> Regex<'static> {
    let mut r = Regex::new();
    let emp = r.empty_string();
    let and1 = r.mk_and(&[emp, emp]);
    r
}
