use bytemuck_derive::{Pod, Zeroable};

use crate::hashcons::VecNode;

#[derive(Pod, Zeroable, Clone, Copy)]
#[repr(transparent)]
pub struct ExprRef(u32);

pub enum Expr<'a> {
    EmptyString,
    NoMatch,
    Byte(u8),
    ByteSet(&'a [u32]),
    Repeat(ExprRef, u32, u32),
    Concat(&'a [ExprRef]),
    Or(&'a [ExprRef]),
    And(&'a [ExprRef]),
}

impl<'a> Expr<'a> {
    fn from_slice(s: &'a [u32]) -> Expr<'a> {
        match s[0] {
            1 => Expr::EmptyString,
            2 => Expr::NoMatch,
            3 => Expr::Byte(s[1] as u8),
            4 => Expr::ByteSet(&s[1..]),
            5 => Expr::Repeat(ExprRef(s[1]), s[2], s[3]),
            6 => Expr::Concat(bytemuck::cast_slice(&s[1..])),
            7 => Expr::Or(bytemuck::cast_slice(&s[1..])),
            8 => Expr::And(bytemuck::cast_slice(&s[1..])),
            _ => panic!("invalid tag: {}", s[0]),
        }
    }
}

impl<'a> VecNode for Expr<'a> {
    type Ref = ExprRef;

    fn wrap_ref(v: u32) -> ExprRef {
        ExprRef(v)
    }

    fn unwrap_ref(r: ExprRef) -> u32 {
        r.0
    }

    fn serialize(&self) -> Vec<u32> {
        fn nary_serialize(tag: u32, es: &[ExprRef]) -> Vec<u32> {
            let mut v = Vec::with_capacity(1 + es.len());
            v.push(tag);
            v.extend_from_slice(bytemuck::cast_slice(es));
            v
        }
        match self {
            Expr::EmptyString => vec![1],
            Expr::NoMatch => vec![2],
            Expr::Byte(b) => vec![3, *b as u32],
            Expr::ByteSet(s) => {
                assert!(s.len() == 256 / 32);
                let mut v = Vec::with_capacity(1 + s.len());
                v.push(4);
                v.extend_from_slice(s);
                v
            }
            Expr::Repeat(e, a, b) => vec![5, e.0, *a, *b],
            Expr::Concat(es) => nary_serialize(6, es),
            Expr::Or(es) => nary_serialize(7, es),
            Expr::And(es) => nary_serialize(8, es),
        }
    }
}
