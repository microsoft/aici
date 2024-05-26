use crate::ast::{Expr, ExprRef, ExprSet};

pub struct Regex {
    exprs: ExprSet,
    state_table: Vec<Vec<ExprRef>>,
}

impl Regex {
    pub fn new() -> Self {
        Regex {
            exprs: ExprSet::new(),
            state_table: vec![],
        }
    }

    pub fn derivative(&mut self, e: ExprRef, b: u8) -> ExprRef {
        let idx = e.as_usize();

        if idx >= self.state_table.len() {
            self.state_table
                .extend((self.state_table.len()..=idx).map(|_| vec![]));
        }
        let vec = &self.state_table[idx];
        if vec.len() > 0 && vec[b as usize].is_valid() {
            return vec[b as usize];
        }

        let d = self.derivative_inner(e, b);

        if self.state_table[idx].len() == 0 {
            self.state_table[idx] = vec![ExprRef::INVALID; 256];
        }
        self.state_table[idx][b as usize] = d;

        d
    }

    fn derivative_inner(&mut self, e: ExprRef, b: u8) -> ExprRef {
        let e = self.exprs.get(e);
        match e {
            Expr::EmptyString | Expr::NoMatch | Expr::ByteSet(_) | Expr::Byte(_) => {
                if e.matches_byte(b) {
                    self.exprs.mk_empty_string()
                } else {
                    self.exprs.mk_no_match()
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
