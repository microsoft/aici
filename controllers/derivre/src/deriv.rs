use crate::{ast::Expr, hashcons::HashCons};

struct Regex {
    exprs: HashCons<Expr<'self>>,
}
