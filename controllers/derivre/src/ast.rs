use bytemuck_derive::{Pod, Zeroable};

use crate::hashcons::VecHashMap;

#[derive(Pod, Zeroable, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ExprRef(pub u32);

impl ExprRef {
    pub const INVALID: ExprRef = ExprRef(0);
    pub fn is_valid(&self) -> bool {
        self.0 != 0
    }
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

pub const BYTE_SET_SIZE: usize = 256 / 32;

pub enum Expr<'a> {
    EmptyString,
    NoMatch,
    Byte(u8),
    ByteSet(&'a [u32]),
    Not(ExprFlags, ExprRef),
    Repeat(ExprFlags, ExprRef, u32, u32),
    Concat(ExprFlags, &'a [ExprRef]),
    Or(ExprFlags, &'a [ExprRef]),
    And(ExprFlags, &'a [ExprRef]),
}

pub enum MatchState {
    Accept,
    Reject,
    Continue,
}

#[derive(Clone, Copy)]
pub struct ExprFlags(u32);
impl ExprFlags {
    const NULLABLE: ExprFlags = ExprFlags(1 << 8);
    const ZERO: ExprFlags = ExprFlags(0);

    pub fn is_nullable(&self) -> bool {
        self.0 & ExprFlags::NULLABLE.0 != 0
    }

    fn from_nullable(nullable: bool) -> Self {
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
enum ExprTag {
    EmptyString = 1,
    NoMatch,
    Byte,
    ByteSet,
    Not,
    Repeat,
    Concat,
    Or,
    And,
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

impl<'a> Expr<'a> {
    pub fn matches_byte(&self, b: u8) -> bool {
        match self {
            Expr::EmptyString => false,
            Expr::NoMatch => false,
            Expr::Byte(b2) => b == *b2,
            Expr::ByteSet(s) => s[(b / 32) as usize] & (1 << (b % 32)) != 0,
            _ => panic!("not a simple expression"),
        }
    }

    fn get_flags(&self) -> ExprFlags {
        match self {
            Expr::EmptyString | Expr::NoMatch | Expr::Byte(_) | Expr::ByteSet(_) => ExprFlags::ZERO,
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

    pub fn classify_state(&self) -> MatchState {
        match self {
            Expr::NoMatch => MatchState::Reject,
            _ if self.nullable() => MatchState::Accept,
            _ => MatchState::Continue,
        }
    }

    fn from_slice(s: &'a [u32]) -> Expr<'a> {
        let flags = ExprFlags(s[0] & !0xff);
        let tag = ExprTag::from_u8((s[0] & 0xff) as u8);
        match tag {
            ExprTag::EmptyString => Expr::EmptyString,
            ExprTag::NoMatch => Expr::NoMatch,
            ExprTag::Byte => Expr::Byte(s[1] as u8),
            ExprTag::ByteSet => Expr::ByteSet(&s[1..]),
            ExprTag::Not => Expr::Not(flags, ExprRef(s[1])),
            ExprTag::Repeat => Expr::Repeat(flags, ExprRef(s[1]), s[2], s[3]),
            ExprTag::Concat => Expr::Concat(flags, bytemuck::cast_slice(&s[1..])),
            ExprTag::Or => Expr::Or(flags, bytemuck::cast_slice(&s[1..])),
            ExprTag::And => Expr::And(flags, bytemuck::cast_slice(&s[1..])),
        }
    }

    fn serialize(&self) -> Vec<u32> {
        fn nary_serialize(tag: u32, es: &[ExprRef]) -> Vec<u32> {
            let mut v = Vec::with_capacity(1 + es.len());
            v.push(tag);
            v.extend_from_slice(bytemuck::cast_slice(es));
            v
        }
        let zf = ExprFlags::ZERO;
        match self {
            Expr::EmptyString => vec![zf.encode(ExprTag::EmptyString)],
            Expr::NoMatch => vec![zf.encode(ExprTag::NoMatch)],
            Expr::Byte(b) => vec![zf.encode(ExprTag::Byte), *b as u32],
            Expr::ByteSet(s) => {
                assert!(s.len() == BYTE_SET_SIZE);
                let mut v = Vec::with_capacity(1 + s.len());
                v.push(zf.encode(ExprTag::ByteSet));
                v.extend_from_slice(s);
                v
            }
            Expr::Not(flags, e) => vec![flags.encode(ExprTag::Not), e.0],
            Expr::Repeat(flags, e, a, b) => vec![flags.encode(ExprTag::Repeat), e.0, *a, *b],
            Expr::Concat(flags, es) => nary_serialize(flags.encode(ExprTag::Concat), es),
            Expr::Or(flags, es) => nary_serialize(flags.encode(ExprTag::Or), es),
            Expr::And(flags, es) => nary_serialize(flags.encode(ExprTag::And), es),
        }
    }
}

pub struct ExprSet {
    empty: ExprRef,
    no_match: ExprRef,
    any_byte: ExprRef,
    any_string: ExprRef,
    non_empty: ExprRef,
    exprs: VecHashMap,
}

impl ExprSet {
    pub fn new() -> Self {
        let mut exprs = VecHashMap::new();
        let empty = ExprRef(exprs.insert(Expr::EmptyString.serialize()));
        assert!(empty.is_valid());
        let no_match = exprs.insert(Expr::NoMatch.serialize());
        let any_byte =
            ExprRef(exprs.insert(Expr::ByteSet(&vec![0xffffffff; BYTE_SET_SIZE]).serialize()));
        let everything =
            exprs.insert(Expr::Repeat(ExprFlags::NULLABLE, any_byte, 0, u32::MAX).serialize());
        let non_empty =
            exprs.insert(Expr::Repeat(ExprFlags::ZERO, any_byte, 1, u32::MAX).serialize());
        ExprSet {
            empty,
            no_match: ExprRef(no_match),
            any_byte,
            any_string: ExprRef(everything),
            non_empty: ExprRef(non_empty),
            exprs,
        }
    }

    pub fn len(&self) -> usize {
        self.exprs.len()
    }

    pub fn bytes(&self) -> usize {
        self.exprs.bytes()
    }

    pub fn mk_empty_string(&mut self) -> ExprRef {
        self.empty
    }

    pub fn mk_no_match(&mut self) -> ExprRef {
        self.no_match
    }

    pub fn mk_any_byte(&mut self) -> ExprRef {
        self.any_byte
    }

    pub fn mk_non_empty(&mut self) -> ExprRef {
        self.non_empty
    }

    pub fn mk_any_string(&mut self) -> ExprRef {
        self.any_string
    }

    pub fn mk_byte(&mut self, b: u8) -> ExprRef {
        self.mk(Expr::Byte(b))
    }

    pub fn mk_byte_set(&mut self, s: &[u32]) -> ExprRef {
        assert!(s.len() == BYTE_SET_SIZE);
        if s.iter().all(|&x| x == 0) {
            return self.no_match;
        }
        self.mk(Expr::ByteSet(s))
    }

    pub fn mk_repeat(&mut self, e: ExprRef, min: u32, max: u32) -> ExprRef {
        if e == self.no_match {
            if min == 0 {
                self.empty
            } else {
                self.no_match
            }
        } else if min == max {
            self.empty
        } else if min + 1 == max {
            e
        } else if min > max {
            self.no_match
        } else {
            let min = if self.is_nullable(e) { 0 } else { min };
            let flags = ExprFlags::from_nullable(min == 0);
            self.mk(Expr::Repeat(flags, e, min, max))
        }
    }

    pub fn mk_star(&mut self, e: ExprRef) -> ExprRef {
        self.mk_repeat(e, 0, u32::MAX)
    }

    pub fn mk_plus(&mut self, e: ExprRef) -> ExprRef {
        self.mk_repeat(e, 1, u32::MAX)
    }

    fn flatten_tag(&self, exp_tag: ExprTag, args: Vec<ExprRef>) -> Vec<ExprRef> {
        let mut i = 0;
        while i < args.len() {
            let tag = self.get_tag(args[i]);
            if tag == exp_tag {
                // ok, we found tag, we can no longer return the original vector
                let mut res = args[0..i].to_vec();
                while i < args.len() {
                    let tag = self.get_tag(args[i]);
                    if tag != exp_tag {
                        res.push(args[i]);
                    } else {
                        res.extend_from_slice(self.get_args(args[i]));
                    }
                    i += 1;
                }
                return res;
            }
            i += 1;
        }
        args
    }

    pub fn mk_or(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        // TODO deal with byte ranges
        args = self.flatten_tag(ExprTag::Or, args);
        args.sort_by_key(|&e| e.0);
        let mut dp = 0;
        let mut prev = self.no_match;
        let mut nullable = false;
        for idx in 0..args.len() {
            let arg = args[idx];
            if arg == prev || arg == self.no_match {
                continue;
            }
            if arg == self.any_string {
                return self.any_string;
            }
            if !nullable && self.is_nullable(arg) {
                nullable = true;
            }
            args[dp] = arg;
            dp += 1;
            prev = arg;
        }
        args.truncate(dp);

        if args.len() == 0 {
            self.no_match
        } else if args.len() == 1 {
            args[0]
        } else {
            let flags = ExprFlags::from_nullable(nullable);
            self.mk(Expr::Or(flags, &args))
        }
    }

    pub fn mk_and(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        args = self.flatten_tag(ExprTag::And, args);
        args.sort_by_key(|&e| e.0);
        let mut dp = 0;
        let mut prev = self.any_string;
        let mut had_empty = false;
        let mut nullable = true;
        for idx in 0..args.len() {
            let arg = args[idx];
            if arg == prev || arg == self.any_string {
                continue;
            }
            if arg == self.no_match {
                return self.no_match;
            }
            if arg == self.empty {
                had_empty = true;
            }
            if nullable && !self.is_nullable(arg) {
                nullable = false;
            }
            args[dp] = arg;
            dp += 1;
            prev = arg;
        }
        args.truncate(dp);

        if args.len() == 0 {
            self.any_string
        } else if args.len() == 1 {
            args[0]
        } else if had_empty {
            if nullable {
                self.empty
            } else {
                self.no_match
            }
        } else {
            let flags = ExprFlags::from_nullable(nullable);
            self.mk(Expr::And(flags, &args))
        }
    }

    pub fn mk_concat(&mut self, mut args: Vec<ExprRef>) -> ExprRef {
        args = self.flatten_tag(ExprTag::Concat, args);
        args.retain(|&e| e != self.empty);
        if args.len() == 0 {
            self.empty
        } else if args.len() == 1 {
            args[0]
        } else if args.iter().any(|&e| e == self.no_match) {
            self.no_match
        } else {
            let flags = ExprFlags::from_nullable(args.iter().all(|&e| self.is_nullable(e)));
            self.mk(Expr::Concat(flags, &args))
        }
    }

    pub fn mk_not(&mut self, e: ExprRef) -> ExprRef {
        if e == self.empty {
            self.non_empty
        } else if e == self.non_empty {
            self.empty
        } else if e == self.any_string {
            self.no_match
        } else if e == self.no_match {
            self.any_string
        } else {
            let n = self.get(e);
            match n {
                Expr::Not(_, e2) => return e2,
                _ => {}
            }
            let flags = ExprFlags::from_nullable(!n.nullable());
            self.mk(Expr::Not(flags, e))
        }
    }

    fn mk(&mut self, e: Expr) -> ExprRef {
        ExprRef(self.exprs.insert(e.serialize()))
    }

    pub fn get(&self, id: ExprRef) -> Expr {
        Expr::from_slice(self.exprs.get(id.0).unwrap())
    }

    fn get_flags(&self, id: ExprRef) -> ExprFlags {
        assert!(id.is_valid());
        ExprFlags(self.exprs.get(id.0).unwrap()[0] & !0xff)
    }

    fn get_tag(&self, id: ExprRef) -> ExprTag {
        assert!(id.is_valid());
        let tag = self.exprs.get(id.0).unwrap()[0] & 0xff;
        ExprTag::from_u8(tag as u8)
    }

    fn get_args(&self, id: ExprRef) -> &[ExprRef] {
        let s = self.exprs.get(id.0).unwrap();
        let tag = ExprTag::from_u8((s[0] & 0xff) as u8);
        match tag {
            ExprTag::Concat | ExprTag::Or | ExprTag::And => bytemuck::cast_slice(&s[1..]),
            _ => panic!("not a n-ary expression"),
        }
    }

    pub fn is_nullable(&self, id: ExprRef) -> bool {
        self.get_flags(id).is_nullable()
    }
}
