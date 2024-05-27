use anyhow::{bail, Result};
use regex_syntax::{
    hir::{self, ClassUnicode, Hir, HirKind},
    Parser,
};

// TODO possibly use Utf8Sequences from regex-syntax crate

use crate::{
    ast::{byteset_256, byteset_from_range, byteset_set, ExprSet},
    ExprRef,
};

struct StackEntry<'a> {
    ast: &'a Hir,
    args: Vec<ExprRef>,
    result_stack_idx: usize,
    result_vec_offset: usize,
}

fn utf8_len_to_max(len: usize) -> u32 {
    match len {
        1 => 0x80 - 1,
        2 => 0x800 - 1,
        3 => 0x10000 - 1,
        4 => 0x10FFFF,
        _ => unreachable!(),
    }
}

impl ExprSet {
    const UTF8_CONT_START: u8 = 0b10_00_0000;
    const UTF8_CONT_END: u8 = 0b10_11_1111;

    // return regex that matches lexicographically ordered utf8 bytes in range [start, end]
    fn utf8_range(&mut self, start: &[u8], end: &[u8]) -> ExprRef {
        assert_eq!(start.len(), end.len());

        let n = start.len();

        let a = start[0];
        let b = end[0];

        if n == 1 {
            return self.mk_byte_set(&byteset_from_range(a, b));
        }

        assert!(a <= b);

        let a_ = self.mk_byte(a);
        let b_ = self.mk_byte(b);

        if a == b {
            let a_next = self.utf8_range(&start[1..], &end[1..]);
            self.mk_concat(vec![a_, a_next])
        } else {
            let start_vec = vec![Self::UTF8_CONT_START; n - 1];
            let end_vec = vec![Self::UTF8_CONT_END; n - 1];
            let a_next = self.utf8_range(&start[1..], &end_vec);
            let mut branches = vec![self.mk_concat(vec![a_, a_next])];
            if b - a > 1 {
                let ab_ = self.mk_byte_set(&byteset_from_range(a + 1, b - 1));
                let ab_next = self.utf8_range(&start_vec, &end_vec);
                branches.push(self.mk_concat(vec![ab_, ab_next]));
            }
            let b_next = self.utf8_range(&start_vec, &end[1..]);
            branches.push(self.mk_concat(vec![b_, b_next]));
            self.mk_or(branches)
        }
    }

    fn handle_unicode_ranges(&mut self, u: &ClassUnicode) -> ExprRef {
        let mut alternatives = Vec::new();
        let mut b_start = [0; 4];
        let mut b_end = [0; 4];
        let mut b_tmp = [0; 4];
        let mut b_tmp2 = [0; 4];
        for range in u.ranges() {
            // println!("   range: {:?}", range);
            let start = range.start();
            let end = range.end();
            assert!(start <= end);
            let mut start_bytes = start.encode_utf8(&mut b_start).as_bytes();
            let end_bytes = end.encode_utf8(&mut b_end).as_bytes();
            while start_bytes.len() < end_bytes.len() {
                let c = utf8_len_to_max(start_bytes.len());
                let tmp_bytes = char::from_u32(c)
                    .unwrap()
                    .encode_utf8(&mut b_tmp)
                    .as_bytes();
                alternatives.push(self.utf8_range(start_bytes, tmp_bytes));
                start_bytes = char::from_u32(c + 1)
                    .unwrap()
                    .encode_utf8(&mut b_tmp2)
                    .as_bytes();
            }
            alternatives.push(self.utf8_range(start_bytes, end_bytes));
        }

        let r = self.mk_or(alternatives);
        // println!("result: {}", self.expr_to_string(r));
        r
    }

    fn from_ast(&mut self, ast: &Hir) -> Result<ExprRef> {
        let mut todo = vec![StackEntry {
            ast,
            args: Vec::new(),
            result_stack_idx: 0,
            result_vec_offset: 0,
        }];
        while let Some(mut node) = todo.pop() {
            let subs = node.ast.kind().subs();
            if subs.len() != node.args.len() {
                assert!(node.args.len() == 0);
                node.args = subs.iter().map(|_| ExprRef::INVALID).collect();
                let result_stack_idx = todo.len();
                todo.push(node);
                for (idx, sub) in subs.iter().enumerate() {
                    todo.push(StackEntry {
                        ast: sub,
                        args: Vec::new(),
                        result_stack_idx,
                        result_vec_offset: idx,
                    });
                }
                continue;
            } else {
                assert!(node.args.iter().all(|&x| x != ExprRef::INVALID));
            }

            let r = match node.ast.kind() {
                HirKind::Empty => ExprRef::EMPTY_STRING,
                HirKind::Literal(bytes) => {
                    let byte_args = bytes.0.iter().map(|b| self.mk_byte(*b)).collect();
                    self.mk_concat(byte_args)
                }
                HirKind::Class(hir::Class::Bytes(ranges)) => {
                    let mut bs = byteset_256();
                    for r in ranges.ranges() {
                        for idx in r.start()..=r.end() {
                            byteset_set(&mut bs, idx as usize);
                        }
                    }
                    self.mk_byte_set(&bs)
                }
                HirKind::Class(hir::Class::Unicode(u)) => self.handle_unicode_ranges(u),
                HirKind::Look(l) => {
                    bail!("lookarounds not supported yet; {:?}", l)
                }
                HirKind::Repetition(r) => {
                    assert!(node.args.len() == 1);
                    // ignoring greedy flag
                    self.mk_repeat(node.args[0], r.min, r.max.unwrap_or(u32::MAX))
                }
                HirKind::Capture(c) => {
                    assert!(node.args.len() == 1);
                    // use (?P<stop>R) as syntax for lookahead
                    if c.name.as_deref() == Some("stop") {
                        self.mk_lookahead(node.args[0], 0)
                    } else {
                        // ignore capture idx/name
                        node.args[0]
                    }
                }
                HirKind::Concat(args) => {
                    assert!(args.len() == node.args.len());
                    self.mk_concat(node.args)
                }
                HirKind::Alternation(args) => {
                    assert!(args.len() == node.args.len());
                    self.mk_or(node.args)
                }
            };

            if todo.is_empty() {
                return Ok(r);
            }

            todo[node.result_stack_idx].args[node.result_vec_offset] = r;
        }
        unreachable!()
    }

    pub fn parse_expr(&mut self, parser: &mut Parser, rx: &str) -> Result<ExprRef> {
        let hir = parser.parse(rx)?;
        self.from_ast(&hir)
    }
}
