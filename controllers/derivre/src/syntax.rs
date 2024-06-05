use anyhow::{bail, Result};
use regex_syntax::{
    hir::{self, ClassUnicode, Hir, HirKind, Look},
    utf8::Utf8Sequences,
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
    allow_start: bool,
    allow_end: bool,
}

impl ExprSet {
    fn handle_unicode_ranges(&mut self, u: &ClassUnicode) -> ExprRef {
        let mut alternatives = Vec::new();

        for range in u.ranges() {
            for seq in Utf8Sequences::new(range.start(), range.end()) {
                let v = seq
                    .as_slice()
                    .iter()
                    .map(|s| self.mk_byte_set(&byteset_from_range(s.start, s.end)))
                    .collect();
                alternatives.push(self.mk_concat(v));
            }
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
            allow_start: true,
            allow_end: true,
        }];
        while let Some(mut node) = todo.pop() {
            let subs = node.ast.kind().subs();
            if subs.len() != node.args.len() {
                assert!(node.args.len() == 0);
                node.args = subs.iter().map(|_| ExprRef::INVALID).collect();
                let result_stack_idx = todo.len();
                let is_concat = matches!(node.ast.kind(), HirKind::Concat(_));
                let derives_start = matches!(
                    node.ast.kind(),
                    HirKind::Alternation(_) | HirKind::Capture(_)
                );
                let allow_start = (derives_start || is_concat) && node.allow_start;
                let allow_end = (derives_start || is_concat) && node.allow_end;
                todo.push(node);
                for (idx, sub) in subs.iter().enumerate() {
                    todo.push(StackEntry {
                        ast: sub,
                        args: Vec::new(),
                        result_stack_idx,
                        result_vec_offset: idx,
                        allow_start: (!is_concat || idx == 0) && allow_start,
                        allow_end: (!is_concat || idx == subs.len() - 1) && allow_end,
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
                // ignore ^ and $ anchors:
                HirKind::Look(Look::Start) if node.allow_start => ExprRef::EMPTY_STRING,
                HirKind::Look(Look::End) if node.allow_end => ExprRef::EMPTY_STRING,
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

    pub fn parse_expr(&mut self, mut parser: Parser, rx: &str) -> Result<ExprRef> {
        let hir = parser.parse(rx)?;
        self.from_ast(&hir)
    }
}
