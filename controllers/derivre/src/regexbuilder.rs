use anyhow::{ensure, Result};
use regex_syntax::ParserBuilder;
use serde::{Deserialize, Serialize};

use crate::{ast::ExprSet, ExprRef, RegexVec};

pub struct RegexBuilder {
    parser_builder: ParserBuilder,
    exprset: ExprSet,
}

#[derive(Serialize, Deserialize)]
pub enum RegexAst {
    And(Vec<RegexAst>),
    Or(Vec<RegexAst>),
    Concat(Vec<RegexAst>),
    Not(Box<RegexAst>),
    EmptyString,
    NoMatch,
    Regex(String),
    #[serde(skip)]
    ExprRef(ExprRef),
}

impl RegexAst {
    fn get_args(&self) -> &[RegexAst] {
        match self {
            RegexAst::And(asts) => asts,
            RegexAst::Or(asts) => asts,
            RegexAst::Concat(asts) => asts,
            RegexAst::Not(ast) => std::slice::from_ref(ast),
            _ => &[],
        }
    }
}
struct MkStackNode<'a> {
    ast: &'a RegexAst,
    trg: usize,
    args: Vec<ExprRef>,
}

impl RegexBuilder {
    pub fn new() -> Self {
        Self {
            parser_builder: ParserBuilder::new(),
            exprset: ExprSet::new(256),
        }
    }

    pub fn mk_regex(&mut self, s: &str) -> Result<ExprRef> {
        let parser = self.parser_builder.build();
        self.exprset.parse_expr(parser, s)
    }

    pub fn mk(&mut self, ast: &RegexAst) -> Result<ExprRef> {
        let mut todo = vec![MkStackNode {
            ast,
            trg: 0,
            args: Vec::new(),
        }];

        while let Some(entry) = todo.pop() {
            let args = entry.ast.get_args();
            if args.len() > 0 && entry.args.len() == 0 {
                let trg = todo.len();
                todo.push(entry);
                for ast in args {
                    todo.push(MkStackNode {
                        ast,
                        trg,
                        args: Vec::new(),
                    });
                }
            } else {
                assert!(entry.args.len() == args.len());
                let new_args = entry.args;
                let r = match ast {
                    RegexAst::Regex(s) => self.mk_regex(s)?,
                    RegexAst::ExprRef(r) => {
                        ensure!(self.exprset.is_valid(*r), "invalid ref");
                        *r
                    }
                    RegexAst::And(_) => self.exprset.mk_and(new_args),
                    RegexAst::Or(_) => self.exprset.mk_or(new_args),
                    RegexAst::Concat(_) => self.exprset.mk_concat(new_args),
                    RegexAst::Not(_) => self.exprset.mk_not(new_args[0]),
                    RegexAst::EmptyString => ExprRef::EMPTY_STRING,
                    RegexAst::NoMatch => ExprRef::NO_MATCH,
                };
                if todo.len() == 0 {
                    return Ok(r);
                }
                todo[entry.trg].args.push(r);
            }
        }

        unreachable!()
    }

    pub fn to_regex_vec(self, rx_list: &[ExprRef]) -> RegexVec {
        RegexVec::new_with_exprset(self.exprset, rx_list)
    }
}

// regex flags; docs copied from regex_syntax crate
impl RegexBuilder {
    /// Enable or disable the Unicode flag (`u`) by default.
    ///
    /// By default this is **enabled**. It may alternatively be selectively
    /// disabled in the regular expression itself via the `u` flag.
    ///
    /// Note that unless `utf8` is disabled (it's enabled by default), a
    /// regular expression will fail to parse if Unicode mode is disabled and a
    /// sub-expression could possibly match invalid UTF-8.
    pub fn unicode(&mut self, unicode: bool) -> &mut Self {
        self.parser_builder.unicode(unicode);
        self
    }

    /// When disabled, translation will permit the construction of a regular
    /// expression that may match invalid UTF-8.
    ///
    /// When enabled (the default), the translator is guaranteed to produce an
    /// expression that, for non-empty matches, will only ever produce spans
    /// that are entirely valid UTF-8 (otherwise, the translator will return an
    /// error).
    pub fn utf8(&mut self, utf8: bool) -> &mut Self {
        self.parser_builder.utf8(utf8);
        self
    }

    /// Enable verbose mode in the regular expression.
    ///
    /// When enabled, verbose mode permits insignificant whitespace in many
    /// places in the regular expression, as well as comments. Comments are
    /// started using `#` and continue until the end of the line.
    ///
    /// By default, this is disabled. It may be selectively enabled in the
    /// regular expression by using the `x` flag regardless of this setting.
    pub fn ignore_whitespace(&mut self, ignore_whitespace: bool) -> &mut Self {
        self.parser_builder.ignore_whitespace(ignore_whitespace);
        self
    }

    /// Enable or disable the case insensitive flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `i` flag.
    pub fn case_insensitive(&mut self, case_insensitive: bool) -> &mut Self {
        self.parser_builder.case_insensitive(case_insensitive);
        self
    }

    /// Enable or disable the "dot matches any character" flag by default.
    ///
    /// By default this is disabled. It may alternatively be selectively
    /// enabled in the regular expression itself via the `s` flag.
    pub fn dot_matches_new_line(&mut self, dot_matches_new_line: bool) -> &mut Self {
        self.parser_builder
            .dot_matches_new_line(dot_matches_new_line);
        self
    }
}
