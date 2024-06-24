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
    pub num_deriv: usize,
    state_table: HashMap<(ExprRef, u8), ExprRef>,
}

impl DerivCache {
    pub fn new() -> Self {
        DerivCache {
            num_deriv: 0,
            state_table: HashMap::default(),
        }
    }

    pub fn derivative(&mut self, exprs: &mut ExprSet, r: ExprRef, b: u8) -> ExprRef {
        exprs.map(
            r,
            &mut self.state_table,
            |r| (r, b),
            |exprs, deriv, r| {
                let e = exprs.get(r);
                self.num_deriv += 1;
                let d = match e {
                    Expr::EmptyString | Expr::NoMatch | Expr::ByteSet(_) | Expr::Byte(_) => {
                        if e.matches_byte(b) {
                            ExprRef::EMPTY_STRING
                        } else {
                            ExprRef::NO_MATCH
                        }
                    }
                    Expr::And(_, _) => exprs.mk_and(deriv),
                    Expr::Or(_, _) => exprs.mk_or(deriv),
                    Expr::Not(_, _) => exprs.mk_not(deriv[0]),
                    Expr::Repeat(_, e, min, max) => {
                        let max = if max == u32::MAX {
                            u32::MAX
                        } else {
                            max.saturating_sub(1)
                        };
                        let tail = exprs.mk_repeat(e, min.saturating_sub(1), max);
                        exprs.mk_concat(vec![deriv[0], tail])
                    }
                    Expr::Concat(_, args) => {
                        let mut or_branches = vec![];
                        let mut args = args.to_vec();
                        for i in 0..args.len() {
                            let nullable = exprs.is_nullable(args[i]);
                            args[i] = deriv[i];
                            or_branches.push(exprs.mk_concat(args[i..].to_vec()));
                            if !nullable {
                                break;
                            }
                        }
                        exprs.mk_or(or_branches)
                    }
                    Expr::Lookahead(_, e, offset) => {
                        if e == ExprRef::EMPTY_STRING {
                            ExprRef::NO_MATCH
                        } else {
                            exprs.mk_lookahead(deriv[0], offset + 1)
                        }
                    }
                };
                debug!(
                    "deriv({}) via {} = {}",
                    exprs.expr_to_string(r),
                    exprs.pp().byte_to_string(b),
                    exprs.expr_to_string(d)
                );
                d
            },
        )
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.state_table.len() * 8 * std::mem::size_of::<isize>()
    }
}
