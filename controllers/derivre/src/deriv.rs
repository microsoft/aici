use std::collections::HashMap;

use crate::ast::{Expr, ExprRef, ExprSet, NextByte};

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
    next_byte_cache: HashMap<ExprRef, NextByte>,
}

impl DerivCache {
    pub fn new(exprs: ExprSet) -> Self {
        DerivCache {
            exprs,
            num_deriv: 0,
            state_table: HashMap::default(),
            next_byte_cache: HashMap::default(),
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
        self.exprs.map(
            r,
            &mut self.state_table,
            |r| (r, b),
            |exprs, deriv, r| {
                let e = exprs.get(r);
                self.num_deriv += 1;
                match e {
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
                    Expr::Repeat(_, _, min, max) => {
                        let max = if max == u32::MAX {
                            u32::MAX
                        } else {
                            max.saturating_sub(1)
                        };
                        let tail = exprs.mk_repeat(deriv[0], min.saturating_sub(1), max);
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
                }
            },
        )
    }

    pub fn next_byte(&mut self, e: ExprRef) -> NextByte {
        if let Some(&nb) = self.next_byte_cache.get(&e) {
            return nb;
        }
        self.next_byte_inner(e);
        self.next_byte_cache[&e]
    }

    fn next_byte_inner(&mut self, r: ExprRef) {
        self.exprs.map(
            r,
            &mut self.next_byte_cache,
            |r| r,
            |exprs, next_byte, r| {
                let e = exprs.get(r);
                match e {
                    Expr::EmptyString => NextByte::ForcedEOI,
                    Expr::NoMatch => NextByte::Dead,
                    Expr::ByteSet(_) => NextByte::SomeBytes,
                    Expr::Byte(b) => NextByte::ForcedByte(b),
                    Expr::And(_, args) => {
                        let mut found = next_byte[0];
                        for child in next_byte.iter().skip(1) {
                            found = found & *child;
                            if found == NextByte::Dead {
                                break;
                            }
                        }
                        match found {
                            NextByte::ForcedByte(b) => {
                                for a in args {
                                    if !exprs.get(*a).matches_byte(b) {
                                        return NextByte::Dead;
                                    }
                                }
                            }
                            NextByte::ForcedEOI => {
                                for a in args {
                                    if !exprs.is_nullable(*a) {
                                        return NextByte::Dead;
                                    }
                                }
                            }
                            NextByte::Dead => {}
                            NextByte::SomeBytes => {}
                        }
                        found
                    }
                    _ => todo!(),
                }
            },
        )
    }
}
