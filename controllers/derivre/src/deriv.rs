use std::collections::HashMap;

use crate::ast::{Expr, ExprRef, ExprSet};

const DEBUG: bool = false;
macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            eprint!("  ");
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone)]
pub struct DerivCache {
    pub exprs: ExprSet,
    pub num_deriv: usize,
    state_table: HashMap<(ExprRef, u8), ExprRef>,
}

impl DerivCache {
    pub fn new(exprs: ExprSet) -> Self {
        DerivCache {
            exprs,
            num_deriv: 0,
            state_table: HashMap::default(),
        }
    }

    pub fn derivative(&mut self, e: ExprRef, b: u8) -> ExprRef {
        let idx = (e, b);
        if let Some(&d) = self.state_table.get(&idx) {
            return d;
        }

        self.derivative_inner(e, b);
        let d = self.state_table[&idx];
        debug!(
            "deriv({}) via {} = {}",
            self.exprs.expr_to_string(e),
            self.exprs.pp().byte_to_string(b),
            self.exprs.expr_to_string(d)
        );

        d
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.exprs.bytes() + self.state_table.len() * 8 * std::mem::size_of::<isize>()
    }

    fn derivative_inner(&mut self, r: ExprRef, b: u8) {
        let mut todo = vec![r];
        while let Some(r) = todo.last() {
            let idx = (*r, b);
            if self.state_table.contains_key(&idx) {
                todo.pop();
                continue;
            }
            let e = self.exprs.get(*r);
            let is_concat = matches!(e, Expr::Concat(_, _));
            let todo_len = todo.len();
            for a in e.args() {
                let brk = is_concat && !self.exprs.is_nullable(*a);
                if self.state_table.contains_key(&(*a, b)) {
                    continue;
                }
                todo.push(*a);
                if brk {
                    break;
                }
            }

            if todo.len() != todo_len {
                continue; // retry children first
            }

            todo.pop(); // pop r

            self.num_deriv += 1;
            let res = match e {
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
                        args[i] = self.state_table[&(args[i], b)];
                    }
                    self.exprs.mk_and(args)
                }
                Expr::Or(_, args) => {
                    let mut args = args.to_vec();
                    for i in 0..args.len() {
                        args[i] = self.state_table[&(args[i], b)];
                    }
                    self.exprs.mk_or(args)
                }
                Expr::Not(_, e) => {
                    let inner = self.state_table[&(e, b)];
                    self.exprs.mk_not(inner)
                }
                Expr::Repeat(_, e, min, max) => {
                    let head = self.state_table[&(e, b)];
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
                        args[i] = self.state_table[&(args[i], b)];
                        or_branches.push(self.exprs.mk_concat(args[i..].to_vec()));
                        if !nullable {
                            break;
                        }
                    }
                    self.exprs.mk_or(or_branches)
                }
                Expr::Lookahead(_, e, offset) => {
                    if e == ExprRef::EMPTY_STRING {
                        ExprRef::NO_MATCH
                    } else {
                        let inner = self.state_table[&(e, b)];
                        self.exprs.mk_lookahead(inner, offset + 1)
                    }
                }
            };
            self.state_table.insert(idx, res);
        }
    }
}
