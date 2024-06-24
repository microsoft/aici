use std::collections::HashMap;

use crate::{
    ast::{Expr, ExprRef, ExprSet, NextByte},
    deriv::DerivCache,
};

#[derive(Clone)]
pub struct NextByteCache {
    next_byte_cache: HashMap<ExprRef, NextByte>,
}

impl NextByteCache {
    pub fn new() -> Self {
        NextByteCache {
            next_byte_cache: HashMap::default(),
        }
    }

    pub fn num_bytes(&self) -> usize {
        self.next_byte_cache.len() * 6 * std::mem::size_of::<isize>()
    }

    pub fn next_byte(
        &mut self,
        exprs: &mut ExprSet,
        deriv: &mut DerivCache,
        r: ExprRef,
    ) -> NextByte {
        exprs.map(
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
                    Expr::And(_, _) => {
                        let mut found = next_byte[0];
                        for child in next_byte.iter().skip(1) {
                            found = found & *child;
                            if found == NextByte::Dead {
                                break;
                            }
                        }
                        // if any branch forces a byte or EOI, the other branches
                        // have to allow it, otherwise the whole thing would
                        // be NO_MATCH
                        found
                    }
                    Expr::Or(_, _) => {
                        let mut found = next_byte[0];
                        for child in next_byte.iter().skip(1) {
                            found = found | *child;
                            if found == NextByte::SomeBytes {
                                break;
                            }
                        }
                        found
                    }
                    Expr::Not(_, _) => NextByte::SomeBytes,
                    Expr::Repeat(_, _, min, _) => {
                        if min == 0 {
                            NextByte::SomeBytes
                        } else {
                            next_byte[0]
                        }
                    }
                    Expr::Concat(_, _) => next_byte[0],
                    Expr::Lookahead(_, _, _) => next_byte[0],
                }
            },
        )
    }
}
