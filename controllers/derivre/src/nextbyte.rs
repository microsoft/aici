use std::collections::HashMap;

use crate::ast::{Expr, ExprRef, ExprSet, NextByte};

#[derive(Clone)]
pub struct NextByteCache {
    next_byte_cache: HashMap<ExprRef, NextByte>,
}

fn next_byte_simple(exprs: &ExprSet, mut r: ExprRef) -> NextByte {
    loop {
        match exprs.get(r) {
            Expr::EmptyString => return NextByte::ForcedEOI,
            Expr::NoMatch => return NextByte::Dead,
            Expr::ByteSet(_) => return NextByte::SomeBytes,
            Expr::Byte(b) => return NextByte::ForcedByte(b),
            Expr::And(_, _) => return NextByte::SomeBytes,
            Expr::Not(_, _) => return NextByte::SomeBytes,
            Expr::Lookahead(_, e, _) => {
                r = e;
            }
            Expr::Repeat(_, arg, min, _) => {
                if min == 0 {
                    return NextByte::SomeBytes;
                } else {
                    r = arg;
                }
            }
            Expr::Concat(_, args) => {
                if exprs.is_nullable(args[0]) {
                    return NextByte::SomeBytes;
                } else {
                    r = args[0];
                }
            }
            Expr::Or(_, _) => return NextByte::SomeBytes,
        }
    }
}

impl NextByteCache {
    pub fn new() -> Self {
        NextByteCache {
            next_byte_cache: HashMap::default(),
        }
    }

    pub fn num_bytes(&self) -> usize {
        self.next_byte_cache.len() * 4 * std::mem::size_of::<isize>()
    }

    pub fn next_byte(&mut self, exprs: &ExprSet, r: ExprRef) -> NextByte {
        if let Some(&found) = self.next_byte_cache.get(&r) {
            return found;
        }
        let next = match exprs.get(r) {
            Expr::Or(_, args) => {
                let mut found = next_byte_simple(exprs, args[0]);
                for child in args.iter().skip(1) {
                    found = found | next_byte_simple(exprs, *child);
                    if found == NextByte::SomeBytes {
                        break;
                    }
                }
                found
            }
            _ => next_byte_simple(exprs, r),
        };
        self.next_byte_cache.insert(r, next);
        next
    }
}
