use aici_abi::wprintln;
use anyhow::Result;
use cfgrammar::{
    yacc::{YaccGrammar, YaccKind},
    TIdx,
};
use lrtable::{from_yacc, Action, Minimiser, StIdx, StateTable};
use regex_automata::{
    dfa::{dense, Automaton},
    util::{primitives::StateID, syntax},
};

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
    let grm = include_bytes!("../c.y");

    let grm = YaccGrammar::new(
        YaccKind::Original(cfgrammar::yacc::YaccOriginalActionKind::NoAction),
        &String::from_utf8_lossy(grm),
    )
    .unwrap();
    let (sgraph, stable) = from_yacc(&grm, Minimiser::Pager).unwrap();

    if true {
        wprintln!("core\n{}\n\n", sgraph.pp(&grm, true));
        for pidx in grm.iter_pidxs() {
            let prod = grm.prod(pidx);
            wprintln!("{:?} -> {}", prod, prod.len());
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

    // for tok in tokens {
    //     let r = psr.parse_lexeme(tok.0, &mut pstack);
    //     wprintln!("t: {:?} {:?} {:?}", tok, grm.token_name(tok), r);
    // }

    let patterns = vec![
        r#"foo"#, //
        r#"fob"#, //
        r#"\w+"#, //
        r#"\d+"#, //
    ];
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .start_kind(regex_automata::dfa::StartKind::Anchored)
                .match_kind(regex_automata::MatchKind::All),
        )
        .syntax(syntax::Config::new().unicode(false).utf8(false))
        .build_many(&patterns)
        .unwrap();

    wprintln!("dfa: {} bytes", dfa.memory_usage());
    //wprintln!("dfa: {:?}", dfa);
    let s = "fooXX";
    let anch = regex_automata::Anchored::Yes;
    let mut state = dfa.universal_start_state(anch).unwrap();
    for b in s.as_bytes() {
        wprintln!("state: {:?} {:?}", state, b);
        let state2 = dfa.next_eoi_state(state);
        if dfa.is_match_state(state2) {
            for idx in 0..dfa.match_len(state2) {
                let pat = patterns[dfa.match_pattern(state2, idx).as_usize()];
                wprintln!("  match: {}", pat);
            }
        } else if dfa.is_dead_state(state) {
            wprintln!("dead");
            break;
        }
        state = dfa.next_state(state, *b);
    }

    Ok(())
}

/*

state: DFA state, set of viable tokens, LR(1) stack

push(byte):
    prev = state
    state = state.next(byte)
    if dead(state):
        tok = matches(prev)
        if tok != white space:
            LR(1) <- tok
        state = state0.next(byte)
        viable = possible_tokens(state) & (viable(LR(1)) | {white space})
    else
        viable = viable & possible_tokens(state)
        if viable is empty
            reject
        else
            continue

*/
