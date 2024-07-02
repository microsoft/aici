use std::{
    hash::Hash,
    ops::{BitAnd, BitOr, RangeInclusive},
};

use crate::{hashcons::VecHashCons, pp::PrettyPrinter};
use bytemuck_derive::{Pod, Zeroable};

#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct ExprRef(pub(crate) u32);

impl ExprRef {
    pub const INVALID: ExprRef = ExprRef(0);
    pub const EMPTY_STRING: ExprRef = ExprRef(1);
    pub const NO_MATCH: ExprRef = ExprRef(2);
    pub const ANY_BYTE: ExprRef = ExprRef(3);
    pub const ANY_STRING: ExprRef = ExprRef(4);
    pub const NON_EMPTY_STRING: ExprRef = ExprRef(5);

    pub fn new(id: u32) -> Self {
        assert!(id != 0, "ExprRef(0) is reserved for invalid reference");
        ExprRef(id)
    }

    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

pub enum Expr<'a> {
    EmptyString,
    NoMatch,
    Byte(u8),
    ByteSet(&'a [u32]),
    Lookahead(ExprFlags, ExprRef, u32),
    Not(ExprFlags, ExprRef),
    Repeat(ExprFlags, ExprRef, u32, u32),
    Concat(ExprFlags, &'a [ExprRef]),
    Or(ExprFlags, &'a [ExprRef]),
    And(ExprFlags, &'a [ExprRef]),
}

#[derive(Clone, Copy)]
pub struct ExprFlags(u32);
impl ExprFlags {
    pub const NULLABLE: ExprFlags = ExprFlags(1 << 8);
    pub const ZERO: ExprFlags = ExprFlags(0);

    pub fn is_nullable(&self) -> bool {
        self.0 & ExprFlags::NULLABLE.0 != 0
    }

    pub fn from_nullable(nullable: bool) -> Self {
        if nullable {
            Self::NULLABLE
        } else {
            Self::ZERO
        }
    }

    fn encode(&self, tag: ExprTag) -> u32 {
        self.0 | tag as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExprTag {
    EmptyString = 1,
    NoMatch,
    Byte,
    ByteSet,
    Lookahead,
    Not,
    Repeat,
    Concat,
    Or,
    And, // has to be last, see below
}

impl ExprTag {
    const MAX_VAL: u8 = ExprTag::And as u8;
    fn from_u8(x: u8) -> Self {
        if x == 0 || x > Self::MAX_VAL {
            panic!("invalid tag: {x}");
        }
        unsafe { std::mem::transmute(x) }
    }
}

#[inline(always)]
pub fn byteset_contains(s: &[u32], b: usize) -> bool {
    s[b / 32] & (1 << (b % 32)) != 0
}

#[inline(always)]
pub fn byteset_set(s: &mut [u32], b: usize) {
    s[b / 32] |= 1 << (b % 32);
}

#[inline(always)]
pub fn byteset_set_range(s: &mut [u32], range: RangeInclusive<u8>) {
    for elt in range {
        byteset_set(s, elt as usize);
    }
}

#[inline(always)]
pub fn byteset_union(s: &mut [u32], other: &[u32]) {
    for i in 0..s.len() {
        s[i] |= other[i];
    }
}

pub fn byteset_256() -> Vec<u32> {
    vec![0u32; 256 / 32]
}

pub fn byteset_from_range(start: u8, end: u8) -> Vec<u32> {
    let mut s = byteset_256();
    byteset_set_range(&mut s, start..=end);
    s
}

impl<'a> Expr<'a> {
    pub fn matches_byte(&self, b: u8) -> bool {
        match self {
            Expr::EmptyString => false,
            Expr::NoMatch => false,
            Expr::Byte(b2) => b == *b2,
            Expr::ByteSet(s) => byteset_contains(s, b as usize),
            _ => panic!("not a simple expression"),
        }
    }

    pub fn args(&self) -> &[ExprRef] {
        match self {
            Expr::Concat(_, es) | Expr::Or(_, es) | Expr::And(_, es) => es,
            Expr::Lookahead(_, e, _) | Expr::Not(_, e) | Expr::Repeat(_, e, _, _) => {
                std::slice::from_ref(e)
            }
            Expr::EmptyString | Expr::NoMatch | Expr::Byte(_) | Expr::ByteSet(_) => &[],
        }
    }

    fn get_flags(&self) -> ExprFlags {
        match self {
            Expr::EmptyString => ExprFlags::NULLABLE,
            Expr::NoMatch | Expr::Byte(_) | Expr::ByteSet(_) => ExprFlags::ZERO,
            Expr::Lookahead(f, _, _) => *f,
            Expr::Not(f, _) => *f,
            Expr::Repeat(f, _, _, _) => *f,
            Expr::Concat(f, _) => *f,
            Expr::Or(f, _) => *f,
            Expr::And(f, _) => *f,
        }
    }

    pub fn nullable(&self) -> bool {
        self.get_flags().is_nullable()
    }

    fn from_slice(s: &'a [u32]) -> Expr<'a> {
        let flags = ExprFlags(s[0] & !0xff);
        let tag = ExprTag::from_u8((s[0] & 0xff) as u8);
        match tag {
            ExprTag::EmptyString => Expr::EmptyString,
            ExprTag::NoMatch => Expr::NoMatch,
            ExprTag::Byte => Expr::Byte(s[1] as u8),
            ExprTag::ByteSet => Expr::ByteSet(&s[1..]),
            ExprTag::Lookahead => Expr::Lookahead(flags, ExprRef::new(s[1]), s[2]),
            ExprTag::Not => Expr::Not(flags, ExprRef::new(s[1])),
            ExprTag::Repeat => Expr::Repeat(flags, ExprRef::new(s[1]), s[2], s[3]),
            ExprTag::Concat => Expr::Concat(flags, bytemuck::cast_slice(&s[1..])),
            ExprTag::Or => Expr::Or(flags, bytemuck::cast_slice(&s[1..])),
            ExprTag::And => Expr::And(flags, bytemuck::cast_slice(&s[1..])),
        }
    }

    fn serialize(&self, trg: &mut VecHashCons) {
        #[inline(always)]
        fn nary_serialize(trg: &mut VecHashCons, tag: u32, es: &[ExprRef]) {
            trg.push_u32(tag);
            trg.push_slice(bytemuck::cast_slice(es));
        }
        let zf = ExprFlags::ZERO;
        match self {
            Expr::EmptyString => trg.push_u32(zf.encode(ExprTag::EmptyString)),
            Expr::NoMatch => trg.push_u32(zf.encode(ExprTag::NoMatch)),
            Expr::Byte(b) => {
                trg.push_slice(&[zf.encode(ExprTag::Byte), *b as u32]);
            }
            Expr::ByteSet(s) => {
                trg.push_u32(zf.encode(ExprTag::ByteSet));
                trg.push_slice(s);
            }
            Expr::Lookahead(flags, e, n) => {
                trg.push_slice(&[flags.encode(ExprTag::Lookahead), e.0, *n]);
            }
            Expr::Not(flags, e) => trg.push_slice(&[flags.encode(ExprTag::Not), e.0]),
            Expr::Repeat(flags, e, a, b) => {
                trg.push_slice(&[flags.encode(ExprTag::Repeat), e.0, *a, *b])
            }
            Expr::Concat(flags, es) => nary_serialize(trg, flags.encode(ExprTag::Concat), es),
            Expr::Or(flags, es) => nary_serialize(trg, flags.encode(ExprTag::Or), es),
            Expr::And(flags, es) => nary_serialize(trg, flags.encode(ExprTag::And), es),
        }
    }
}

#[derive(Clone)]
pub struct ExprSet {
    exprs: VecHashCons,
    pub(crate) alphabet_size: usize,
    pub(crate) alphabet_words: usize,
    pub(crate) cost: usize,
    pp: PrettyPrinter,
    pub(crate) optimize: bool,
}

impl ExprSet {
    pub fn new(alphabet_size: usize) -> Self {
        let exprs = VecHashCons::new();
        let alphabet_words = (alphabet_size + 31) / 32;
        let mut r = ExprSet {
            exprs,
            alphabet_size,
            alphabet_words,
            cost: 0,
            pp: PrettyPrinter::new_simple(alphabet_size),
            optimize: true,
        };

        let id = r.exprs.insert(&[]);
        assert!(id == 0);
        let inserts = vec![
            (r.mk(Expr::EmptyString), ExprRef::EMPTY_STRING),
            (r.mk(Expr::NoMatch), ExprRef::NO_MATCH),
            (
                r.mk(Expr::ByteSet(&vec![0xffffffff; alphabet_words])),
                ExprRef::ANY_BYTE,
            ),
            (
                r.mk(Expr::Repeat(
                    ExprFlags::NULLABLE,
                    ExprRef::ANY_BYTE,
                    0,
                    u32::MAX,
                )),
                ExprRef::ANY_STRING,
            ),
            (
                r.mk(Expr::Repeat(
                    ExprFlags::ZERO,
                    ExprRef::ANY_BYTE,
                    1,
                    u32::MAX,
                )),
                ExprRef::NON_EMPTY_STRING,
            ),
        ];

        for (x, y) in inserts {
            assert!(x == y, "id: {x:?}, expected: {y:?}");
        }

        r
    }

    pub fn set_pp(&mut self, pp: PrettyPrinter) {
        self.pp = pp;
    }

    pub fn pp(&self) -> &PrettyPrinter {
        &self.pp
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    pub(crate) fn disable_optimizations(&mut self) {
        self.optimize = false;
    }

    pub fn expr_to_string(&self, id: ExprRef) -> String {
        self.pp.expr_to_string(&self, id)
    }

    pub fn expr_to_string_with_info(&self, id: ExprRef) -> String {
        let mut r = self.expr_to_string(id);
        r.push_str(&self.pp.alphabet_info());
        r
    }

    pub fn alphabet_size(&self) -> usize {
        self.alphabet_size
    }

    pub fn alphabet_words(&self) -> usize {
        self.alphabet_words
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.exprs.len()
    }

    pub fn num_bytes(&self) -> usize {
        self.exprs.num_bytes()
    }

    // When called outside ctor, one should also call self.pay()
    pub(crate) fn mk(&mut self, e: Expr) -> ExprRef {
        self.exprs.start_insert();
        e.serialize(&mut self.exprs);
        ExprRef(self.exprs.finish_insert())
    }

    pub fn get(&self, id: ExprRef) -> Expr {
        Expr::from_slice(self.exprs.get(id.0))
    }

    pub fn is_valid(&self, id: ExprRef) -> bool {
        id.is_valid() && self.exprs.is_valid(id.0)
    }

    fn lookahead_len_inner(&self, e: ExprRef) -> Option<usize> {
        match self.get(e) {
            Expr::Lookahead(_, ExprRef::EMPTY_STRING, n) => Some(n as usize),
            _ => None,
        }
    }

    pub fn lookahead_len(&self, e: ExprRef) -> Option<usize> {
        match self.get(e) {
            Expr::Or(_, args) => args
                .iter()
                .filter_map(|&arg| self.lookahead_len_inner(arg))
                .min(),
            _ => self.lookahead_len_inner(e),
        }
    }

    fn possible_lookahead_len_inner(&self, e: ExprRef) -> usize {
        match self.get(e) {
            Expr::Lookahead(_, _, n) => n as usize,
            _ => 0,
        }
    }

    pub fn possible_lookahead_len(&self, e: ExprRef) -> usize {
        match self.get(e) {
            Expr::Or(_, args) => args
                .iter()
                .map(|&arg| self.possible_lookahead_len_inner(arg))
                .max()
                .unwrap_or(0),
            _ => self.possible_lookahead_len_inner(e),
        }
    }

    fn get_flags(&self, id: ExprRef) -> ExprFlags {
        assert!(id.is_valid());
        if id == ExprRef::EMPTY_STRING {
            return ExprFlags::NULLABLE;
        }
        ExprFlags(self.exprs.get(id.0)[0] & !0xff)
    }

    pub fn get_tag(&self, id: ExprRef) -> ExprTag {
        assert!(id.is_valid());
        let tag = self.exprs.get(id.0)[0] & 0xff;
        ExprTag::from_u8(tag as u8)
    }

    pub fn get_args(&self, id: ExprRef) -> &[ExprRef] {
        let s = self.exprs.get(id.0);
        let tag = ExprTag::from_u8((s[0] & 0xff) as u8);
        match tag {
            ExprTag::Concat | ExprTag::Or | ExprTag::And => bytemuck::cast_slice(&s[1..]),
            ExprTag::Not | ExprTag::Repeat | ExprTag::Lookahead => bytemuck::cast_slice(&s[1..2]),
            ExprTag::EmptyString | ExprTag::NoMatch | ExprTag::Byte | ExprTag::ByteSet => &[],
        }
    }

    pub fn is_nullable(&self, id: ExprRef) -> bool {
        self.get_flags(id).is_nullable()
    }

    #[inline(always)]
    pub fn map<K: Eq + PartialEq + Hash, V: Clone>(
        &mut self,
        r: ExprRef,
        cache: &mut std::collections::HashMap<K, V>,
        mk_key: impl Fn(ExprRef) -> K,
        mut process: impl FnMut(&mut ExprSet, Vec<V>, ExprRef) -> V,
    ) -> V {
        if let Some(d) = cache.get(&mk_key(r)) {
            return d.clone();
        }

        let mut todo = vec![r];
        while let Some(r) = todo.last() {
            let r = *r;
            let idx = mk_key(r);
            if cache.contains_key(&idx) {
                todo.pop();
                continue;
            }
            let e = self.get(r);
            let is_concat = matches!(e, Expr::Concat(_, _));
            let todo_len = todo.len();
            let eargs = e.args();
            let mut mapped = Vec::with_capacity(eargs.len());
            for a in eargs {
                let a = *a;
                let brk = is_concat && !self.is_nullable(a);
                if let Some(v) = cache.get(&mk_key(a)) {
                    mapped.push(v.clone());
                } else {
                    todo.push(a);
                }
                if brk {
                    break;
                }
            }

            if todo.len() != todo_len {
                continue; // retry children first
            }

            todo.pop(); // pop r

            let v = process(self, mapped, r);
            cache.insert(idx, v);
        }
        cache[&mk_key(r)].clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextByte {
    /// Transition via any other byte, or EOI leads to a dead state.
    ForcedByte(u8),
    /// Transition via any byte leads to a dead state but EOI is possible.
    ForcedEOI,
    /// Transition via some bytes *may be* possible.
    SomeBytes,
    /// The current state is dead.
    /// Should be only true for NO_MATCH.
    Dead,
}

impl BitAnd for NextByte {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            if self == NextByte::SomeBytes {
                other
            } else if other == NextByte::SomeBytes {
                self
            } else {
                NextByte::Dead
            }
        }
    }
}

impl BitOr for NextByte {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            if self == NextByte::Dead {
                other
            } else if other == NextByte::Dead {
                self
            } else {
                NextByte::SomeBytes
            }
        }
    }
}
