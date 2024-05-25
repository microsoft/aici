use crate::{
    ast::{Expr, ExprRef},
    hashcons::HashCons,
};

struct Regex<'a> {
    exprs: HashCons<Expr<'a>>,
}

impl<'a> Regex<'a> {
    pub fn new() -> Self {
        Regex {
            exprs: HashCons::new(),
        }
    }

    pub fn empty_string(&mut self) -> ExprRef {
        let d = self.exprs.serialize(&Expr::EmptyString);
        self.exprs.insert(d)
    }

    pub fn mk_and<'b>(&mut self, es: &'b [ExprRef]) -> ExprRef
    {
        let d = self.exprs.serialize(&Expr::And(es));
        self.exprs.insert(d)
    }
}

fn foo() -> Regex<'static> {
    let mut r = Regex::new();
    let emp = r.empty_string();
    let _and1 = r.mk_and(&[emp, emp]);
    r
}
