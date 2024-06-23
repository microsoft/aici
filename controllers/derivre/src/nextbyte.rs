use std::collections::HashMap;

use crate::{
    ast::{Expr, ExprRef, ExprSet, NextByte},
    deriv::DerivCache,
};

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
                                for a in args.to_vec() {
                                    if deriv.derivative(exprs, a, b) == ExprRef::NO_MATCH {
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
