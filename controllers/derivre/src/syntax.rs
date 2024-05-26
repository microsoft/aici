use anyhow::{bail, ensure, Result};
use regex_syntax::ast::Ast;

use crate::{ast::ExprSet, ExprRef};

impl ExprSet {
    fn from_ast(&mut self, ast: &Ast) -> Result<ExprRef> {
        match ast {
            Ast::Empty(_) => Ok(self.mk_empty_string()),
            Ast::Flags(_) => bail!("flags not supported"),
            Ast::Literal(l) => {
                ensure!((l.c as u32) < 0x80, "only ASCII supported right now");
                Ok(self.mk_byte(l.c as u8))
            }
            Ast::Dot(_) => todo!(),
            Ast::Assertion(_) => todo!(),
            Ast::ClassUnicode(_) => todo!(),
            Ast::ClassPerl(_) => todo!(),
            Ast::ClassBracketed(_) => todo!(),
            Ast::Repetition(_) => todo!(),
            Ast::Group(_) => todo!(),
            Ast::Alternation(_) => todo!(),
            Ast::Concat(_) => todo!(),
        }
    }

    pub fn expr_from_str(&mut self, rx: &str) -> Result<ExprRef> {
        let mut parser = regex_syntax::ast::parse::Parser::new();
        let parsed = parser.parse_with_comments(rx)?;
        self.from_ast(&parsed.ast)
    }
}
