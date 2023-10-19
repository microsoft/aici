use anyhow::Result;
use cfgrammar::{
    yacc::{YaccGrammar, YaccKind},
    TIdx,
};
use clap::Parser;
use gvm_tokenizers::find_tokenizer;
use lrtable::{from_yacc, Action, Minimiser, StIdx, StateTable};

#[derive(Parser)]
struct Cli {
    /// Tokenizer to use; try -t list to see options
    #[arg(short, long, default_value = "llama")]
    tokenizer: String,
}

type StorageT = u32;

struct ParserState<'a> {
    grm: &'a YaccGrammar<StorageT>,
    stable: &'a StateTable<StorageT>,
}

type PStack<StorageT> = Vec<StIdx<StorageT>>; // Parse stack

#[derive(Debug, Clone, Copy)]
enum ParseResult {
    Accept,
    Error,
    Continue,
}

impl<'a> ParserState<'a> {
    fn parse_lexeme(&self, lexeme: StorageT, pstack: &mut PStack<StorageT>) -> ParseResult {
        loop {
            let stidx = *pstack.last().unwrap();
            let la_tidx = TIdx(lexeme);

            match self.stable.action(stidx, la_tidx) {
                Action::Reduce(pidx) => {
                    let ridx = self.grm.prod_to_rule(pidx);
                    let pop_idx = pstack.len() - self.grm.prod(pidx).len();
                    pstack.drain(pop_idx..);
                    let prior = *pstack.last().unwrap();
                    pstack.push(self.stable.goto(prior, ridx).unwrap());
                }
                Action::Shift(state_id) => {
                    pstack.push(state_id);
                    return ParseResult::Continue;
                }
                Action::Accept => {
                    // only happens when lexeme is EOF
                    return ParseResult::Accept;
                }
                Action::Error => {
                    return ParseResult::Error;
                }
            }
        }
    }
}

pub fn cfg_test() -> Result<()> {
    let grm = r#"
%start Expr
%%
Expr:
      Expr '+' Term
    | Term
    ;

Term:
      Term '*' Q Factor
    | Factor
    ;

Q: /* nothing */ | 'Q' ;

Factor:
      '(' Expr ')'
      | '0'
    | '1'
    | '2'
    | '3'
    ;
%%
    "#;
    let cli = Cli::parse();

    let mut tokenizer = find_tokenizer(&cli.tokenizer)?;
    tokenizer.load();

    println!("loaded tokenizer: {}", tokenizer.name);

    let grm = YaccGrammar::new(
        YaccKind::Original(cfgrammar::yacc::YaccOriginalActionKind::NoAction),
        grm,
    )
    .unwrap();
    let (sgraph, stable) = from_yacc(&grm, Minimiser::Pager).unwrap();

    if false {
        println!("core\n{}\n\n", sgraph.pp(&grm, true));
        for pidx in grm.iter_pidxs() {
            let prod = grm.prod(pidx);
            println!("{:?} -> {}", prod, prod.len());
        }
    }

    let mut pstack = Vec::new();
    pstack.push(stable.start_state());
    let psr = ParserState {
        grm: &grm,
        stable: &stable,
    };

    let s = "(0+1)*Q2";
    let mut tokens = s
        .char_indices()
        .map(|(index, ch)| &s[index..index + ch.len_utf8()])
        .map(|chstr| grm.token_idx(chstr).unwrap())
        .collect::<Vec<_>>();
    tokens.push(grm.eof_token_idx());

    for tok in tokens {
        let r = psr.parse_lexeme(tok.0, &mut pstack);
        println!("t: {:?} {:?} {:?}", tok, grm.token_name(tok), r);
    }

    // println!("loaded grammar\n{}", sgraph.pp(&grm, false));

    Ok(())
}
