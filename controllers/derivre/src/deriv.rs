use std::collections::HashMap;

use crate::ast::{Expr, ExprRef, ExprSet};

pub struct DerivCache {
    exprs: ExprSet,
    state_table: HashMap<(ExprRef, u8), ExprRef>,
}

impl DerivCache {
    pub fn new() -> Self {
        DerivCache {
            exprs: ExprSet::new(256),
            state_table: HashMap::default(),
        }
    }

    pub fn derivative(&mut self, e: ExprRef, b: u8) -> ExprRef {
        let idx = (e, b);
        if let Some(&d) = self.state_table.get(&idx) {
            return d;
        }

        let d = self.derivative_inner(e, b);

        self.state_table.insert(idx, d);

        d
    }

    pub fn get_expr(&self, e: ExprRef) -> Expr {
        self.exprs.get(e)
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn bytes(&self) -> usize {
        self.exprs.bytes() + self.state_table.len() * 8 * std::mem::size_of::<isize>()
    }

    fn derivative_inner(&mut self, e: ExprRef, b: u8) -> ExprRef {
        let e = self.exprs.get(e);
        match e {
            Expr::EmptyString | Expr::NoMatch | Expr::ByteSet(_) | Expr::Byte(_) => {
                if e.matches_byte(b) {
                    ExprRef::EMPTY_STRING
                } else {
                    ExprRef::NO_MATCH
                }
            }
            Expr::And(_, args) => {
                let mut args = args.to_vec();
                for i in 0..args.len() {
                    args[i] = self.derivative(args[i], b);
                }
                self.exprs.mk_and(args)
            }
            Expr::Or(_, args) => {
                let mut args = args.to_vec();
                for i in 0..args.len() {
                    args[i] = self.derivative(args[i], b);
                }
                self.exprs.mk_or(args)
            }
            Expr::Not(_, e) => {
                let inner = self.derivative(e, b);
                self.exprs.mk_not(inner)
            }
            Expr::Repeat(_, e, min, max) => {
                let head = self.derivative(e, b);
                let max = if max == u32::MAX {
                    u32::MAX
                } else {
                    max.saturating_sub(1)
                };
                let tail = self.exprs.mk_repeat(e, min.saturating_sub(1), max);
                self.exprs.mk_concat(vec![head, tail])
            }
            Expr::Concat(_, args) => {
                let mut args = args.to_vec();
                let mut or_branches = vec![];
                for i in 0..args.len() {
                    let nullable = self.exprs.is_nullable(args[i]);
                    args[i] = self.derivative(args[i], b);
                    or_branches.push(self.exprs.mk_concat(args[i..].to_vec()));
                    if !nullable {
                        break;
                    }
                }
                self.exprs.mk_or(or_branches)
            }
        }
    }
}
