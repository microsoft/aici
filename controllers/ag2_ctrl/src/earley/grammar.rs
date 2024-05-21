use std::fmt::Debug;

use aici_abi::{svob::SimpleVob, toktree::SpecialToken};

use super::{lex::LexemeIdx, ByteSet};
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymIdx(u32);

impl Symbol {
    fn is_terminal(&self) -> bool {
        self.is_lexeme_terminal() || self.is_model_variable()
    }
    fn is_lexeme_terminal(&self) -> bool {
        self.lexeme.is_some()
    }
    fn is_model_variable(&self) -> bool {
        self.props.model_variable.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModelVariable {
    SpecialToken(SpecialToken),
    ActiveRoleEnd,
    Other(String),
}

impl ModelVariable {
    pub fn eos_token() -> Self {
        ModelVariable::SpecialToken(SpecialToken::EndOfSentence)
    }

    #[allow(dead_code)]
    pub fn to_string(&self) -> String {
        match self {
            ModelVariable::ActiveRoleEnd => "active_role_end".to_string(),
            ModelVariable::SpecialToken(SpecialToken::EndOfSentence) => "eos_token".to_string(),
            ModelVariable::SpecialToken(SpecialToken::BeginningOfSentence) => {
                "bos_token".to_string()
            }
            ModelVariable::SpecialToken(s) => format!("{:?}", s),
            ModelVariable::Other(s) => s.clone(),
        }
    }

    pub fn from_string(s: &str) -> Self {
        match s {
            "active_role_end" => ModelVariable::ActiveRoleEnd,
            "eos_token" => ModelVariable::SpecialToken(SpecialToken::EndOfSentence),
            "bos_token" => ModelVariable::SpecialToken(SpecialToken::BeginningOfSentence),
            _ => ModelVariable::Other(s.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SymbolProps {
    pub max_tokens: usize,
    pub commit_point: bool,
    pub capture_name: Option<String>,
    pub hidden: bool,
    pub model_variable: Option<ModelVariable>,
    pub temperature: f32,
}

impl Default for SymbolProps {
    fn default() -> Self {
        SymbolProps {
            commit_point: false,
            hidden: false,
            max_tokens: usize::MAX,
            model_variable: None,
            capture_name: None,
            temperature: 0.0,
        }
    }
}

impl SymbolProps {
    /// Special nodes can't be removed in grammar optimizations
    pub fn is_special(&self) -> bool {
        self.commit_point
            || self.hidden
            || self.max_tokens < usize::MAX
            || self.capture_name.is_some()
    }
}

struct Symbol {
    idx: SymIdx,
    name: String,
    lexeme: Option<LexemeIdx>,
    rules: Vec<Rule>,
    props: SymbolProps,
}

struct Rule {
    lhs: SymIdx,
    rhs: Vec<SymIdx>,
}

impl Rule {
    fn lhs(&self) -> SymIdx {
        self.lhs
    }
}

enum SymName {
    Name(String),
    Byte(u8),
}

impl SymName {
    fn from(name: &str, bytes: Option<&ByteSet>) -> Self {
        if let Some(bytes) = bytes {
            if let Some(b) = bytes.single_byte() {
                return SymName::Byte(b);
            }
        }
        SymName::Name(name.to_string())
    }
}

pub struct Grammar {
    symbols: Vec<Symbol>,
    symbol_by_name: FxHashMap<String, SymIdx>,
    byte_terminals: FxHashMap<ByteSet, SymIdx>,
    model_variables: FxHashMap<String, SymIdx>,
}

impl Grammar {
    pub fn new() -> Self {
        let mut r = Grammar {
            symbols: vec![],
            symbol_by_name: FxHashMap::default(),
            byte_terminals: FxHashMap::default(),
            model_variables: FxHashMap::default(),
        };
        let _ = r.symbol("_start");
        r
    }

    pub fn start(&self) -> SymIdx {
        self.symbols[0].idx
    }

    fn sym_data(&self, sym: SymIdx) -> &Symbol {
        &self.symbols[sym.0 as usize]
    }

    fn sym_data_mut(&mut self, sym: SymIdx) -> &mut Symbol {
        &mut self.symbols[sym.0 as usize]
    }

    pub fn add_rule(&mut self, lhs: SymIdx, rhs: Vec<SymIdx>) {
        assert!(!self.sym_data(lhs).is_terminal());
        let sym = self.sym_data_mut(lhs);
        sym.rules.push(Rule { lhs, rhs });
    }

    pub fn model_variable(&mut self, name: &str) -> SymIdx {
        match self.model_variables.get(name) {
            Some(sym) => *sym,
            None => {
                let sym = self.fresh_symbol(format!("M:{}", name).as_str());
                self.sym_data_mut(sym).props.model_variable =
                    Some(ModelVariable::from_string(name));
                self.model_variables.insert(name.to_string(), sym);
                sym
            }
        }
    }

    pub fn terminal(&mut self, bytes: &ByteSet) -> SymIdx {
        match self.byte_terminals.get(bytes) {
            Some(sym) => *sym,
            None => {
                let mut name = format!("T:{}", bytes);
                if name.len() > 40 {
                    name = format!("T@{}", self.byte_terminals.len());
                }
                let sym = self.fresh_symbol(&name);
                self.sym_data_mut(sym).bytes = Some(bytes.clone());
                self.byte_terminals.insert(bytes.clone(), sym);
                sym
            }
        }
    }

    pub fn sym_name(&self, sym: SymIdx) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    fn rule_to_string(&self, rule: &Rule, dot: Option<usize>) -> String {
        let ldata = self.sym_data(rule.lhs());
        let dot_data = rule
            .rhs
            .get(dot.unwrap_or(0))
            .map(|s| &self.sym_data(*s).props);
        rule_to_string(
            self.sym_name(rule.lhs()),
            rule.rhs
                .iter()
                .map(|s| {
                    let d = self.sym_data(*s);
                    SymName::from(&d.name, d.bytes.as_ref())
                })
                .collect(),
            dot,
            &ldata.props,
            dot_data,
        )
    }

    fn copy_from(&mut self, other: &Grammar, sym: SymIdx) -> SymIdx {
        let sym_data = other.sym_data(sym);
        let r = if sym_data.is_lexeme_terminal() {
            self.terminal(sym_data.bytes.as_ref().unwrap())
        } else {
            self.symbol(&sym_data.name)
        };
        self.sym_data_mut(r).props = sym_data.props.clone();
        r
    }

    fn rule_shape(&self, r: &Rule) -> Vec<Option<SymIdx>> {
        let mut shape = Vec::new();
        let mut had_term = false;
        for s in &r.rhs {
            let sym = self.sym_data(*s);
            if !had_term && sym.is_lexeme_terminal() {
                had_term = true;
                shape.push(None);
            } else {
                shape.push(Some(*s));
            }
        }
        shape
    }

    fn collapse_terminals(&self) -> Self {
        let mut outp = Grammar::new();
        for sym in &self.symbols {
            if sym.rules.is_empty() {
                continue;
            }
            let mut rules_by_shape = FxHashMap::default();
            for rule in &sym.rules {
                rules_by_shape
                    .entry(self.rule_shape(rule))
                    .or_insert_with(Vec::new)
                    .push(rule);
            }
            let lhs = outp.copy_from(self, sym.idx);
            for (shape, rules) in &rules_by_shape {
                let rhs = rules[0]
                    .rhs
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        if shape[i].is_none() {
                            let terminals = rules
                                .iter()
                                .map(|r| self.sym_data(r.rhs[i]).bytes.clone().unwrap());
                            outp.terminal(&ByteSet::from_sum(terminals))
                        } else {
                            outp.copy_from(self, *s)
                        }
                    })
                    .collect();
                outp.add_rule(lhs, rhs);
            }
        }
        outp
    }

    fn rename(&mut self) {
        let name_repl = vec![("zero_or_more", "z"), ("one_or_more", "o")];
        for sym in &mut self.symbols {
            for (from, to) in &name_repl {
                if sym.name.starts_with(from) {
                    sym.name = format!("{}_{}", to, &sym.name[from.len()..]);
                }
            }
        }
        self.symbol_by_name = self
            .symbols
            .iter()
            .map(|s| (s.name.clone(), s.idx))
            .collect();
        assert!(self.symbols.len() == self.symbol_by_name.len());
    }

    fn expand_shortcuts(&self) -> Self {
        let mut use_count = vec![0; self.symbols.len()];
        for sym in &self.symbols {
            for r in sym.rules.iter() {
                for s in &r.rhs {
                    use_count[s.0 as usize] += 1;
                }
            }
        }

        let mut repl = FxHashMap::default();
        for sym in &self.symbols {
            // don't inline special symbols (commit points, captures, ...) or start symbol
            if sym.idx == self.start() || sym.props.is_special() {
                continue;
            }
            if sym.rules.len() == 1
                && (use_count[sym.idx.0 as usize] == 1 || sym.rules[0].rhs.len() == 1)
            {
                // eliminate sym.idx
                repl.insert(sym.idx, sym.rules[0].rhs.clone());
            }
        }

        // fix-point expand the mapping
        // TODO union-find?
        loop {
            let to_change = repl
                .iter()
                .filter_map(|(lhs, rhs)| {
                    let rhs2 = rhs
                        .iter()
                        .flat_map(|s| repl.get(s).cloned().unwrap_or_else(|| vec![*s]))
                        .collect::<Vec<_>>();
                    assert!(rhs2.iter().all(|s| *s != *lhs), "cyclic?");
                    if *rhs != rhs2 {
                        Some((*lhs, rhs2))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if to_change.is_empty() {
                break;
            }
            for (lhs, rhs) in to_change {
                repl.insert(lhs, rhs);
            }
        }

        let mut outp = Grammar::new();
        for sym in &self.symbols {
            if repl.contains_key(&sym.idx) {
                continue;
            }
            let lhs = outp.copy_from(self, sym.idx);
            for rule in &sym.rules {
                let rhs = rule
                    .rhs
                    .iter()
                    .flat_map(|s| repl.get(s).cloned().unwrap_or_else(|| vec![*s]))
                    .map(|s| outp.copy_from(self, s))
                    .collect();
                outp.add_rule(lhs, rhs);
            }
        }
        outp
    }

    pub fn optimize(&self) -> Self {
        let mut r = self
            .expand_shortcuts()
            .collapse_terminals()
            .expand_shortcuts();
        r.rename();
        r
    }

    pub fn compile(&self) -> CGrammar {
        CGrammar::from_grammar(self)
    }

    pub fn apply_props(&mut self, sym: SymIdx, mut props: SymbolProps) {
        let sym = self.sym_data_mut(sym);
        assert!(props.model_variable.is_none());
        props.model_variable = sym.props.model_variable.clone();
        if props.is_special() {
            assert!(!sym.is_terminal(), "special terminal");
        }
        assert!(
            !(!props.commit_point && props.hidden),
            "hidden on non-commit_point"
        );
        sym.props = props;
    }

    pub fn fresh_symbol(&mut self, name0: &str) -> SymIdx {
        let mut name = name0.to_string();
        let mut idx = 2;
        while self.symbol_by_name.contains_key(&name) {
            name = format!("{}#{}", name0, idx);
            idx += 1;
        }

        let idx = SymIdx(self.symbols.len() as u32);
        self.symbols.push(Symbol {
            name: name.clone(),
            bytes: None,
            idx,
            rules: vec![],
            props: SymbolProps::default(),
        });
        self.symbol_by_name.insert(name, idx);
        idx
    }

    pub fn symbol(&mut self, name: &str) -> SymIdx {
        match self.symbol_by_name.get(name) {
            Some(idx) => *idx,
            None => self.fresh_symbol(name),
        }
    }
}

impl Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Grammar:")?;
        for sym in &self.symbols {
            match sym.bytes {
                Some(ref bytes) if sym.name.starts_with("T@") => {
                    writeln!(f, "{} := {}", sym.name, bytes)?
                }
                _ => {}
            }
        }
        let mut num_term = 0;
        let mut num_rules = 0;
        let mut num_non_term = 0;
        for sym in &self.symbols {
            if sym.is_terminal() {
                num_term += 1;
            } else {
                num_non_term += 1;
                num_rules += sym.rules.len();
            }
            for rule in &sym.rules {
                writeln!(f, "{}", self.rule_to_string(rule, None))?;
            }
        }
        writeln!(
            f,
            "stats: {} terminals; {} non-terminals with {} rules\n",
            num_term, num_non_term, num_rules
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CSymIdx(u16);

impl CSymIdx {
    pub const NULL: CSymIdx = CSymIdx(0);

    pub fn as_index(&self) -> usize {
        self.0 as usize
    }
}

pub trait SimpleHash {
    fn simple_hash(&self) -> u32;

    fn mask64(&self) -> u64 {
        1 << (self.simple_hash() & 63)
    }

    fn mask32(&self) -> u32 {
        1 << (self.simple_hash() & 31)
    }
}

impl SimpleHash for CSymIdx {
    fn simple_hash(&self) -> u32 {
        (self.0 as u32).wrapping_mul(79667123)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleIdx(u32);

impl RuleIdx {
    // pub const NULL: RuleIdx = RuleIdx(0);

    pub fn from_index(idx: u32) -> Self {
        RuleIdx(idx)
    }

    pub fn as_index(&self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone)]
pub struct CSymbol {
    pub idx: CSymIdx,
    pub name: String,
    pub is_terminal: bool,
    pub is_nullable: bool,
    pub props: SymbolProps,
    pub rules: Vec<RuleIdx>,
    pub sym_flags: SymFlags,
}

#[derive(Clone, Copy)]
pub struct SymFlags(u8);

impl SymFlags {
    const COMMIT_POINT: u8 = 1 << 0;
    const HIDDEN: u8 = 1 << 2;
    const CAPTURE: u8 = 1 << 3;

    fn from_csymbol(sym: &CSymbol) -> Self {
        let mut flags = 0;
        if sym.props.commit_point {
            flags |= Self::COMMIT_POINT;
        }
        if sym.props.hidden {
            flags |= Self::HIDDEN;
        }
        if sym.props.capture_name.is_some() {
            flags |= Self::CAPTURE;
        }
        SymFlags(flags)
    }

    #[inline(always)]
    pub fn commit_point(&self) -> bool {
        self.0 & Self::COMMIT_POINT != 0
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub fn hidden(&self) -> bool {
        self.0 & Self::HIDDEN != 0
    }

    #[inline(always)]
    pub fn capture(&self) -> bool {
        self.0 & Self::CAPTURE != 0
    }
}

#[derive(Clone)]
pub struct CGrammar {
    start_symbol: CSymIdx,
    terminals: Vec<ByteSet>,
    last_single_byte_terminal: usize,
    symbols: Vec<CSymbol>,
    rules: Vec<CSymIdx>,
    rule_idx_to_sym_idx: Vec<CSymIdx>,
    rule_idx_to_sym_flags: Vec<SymFlags>,
    terminals_by_byte: Vec<SimpleVob>,
}

const RULE_SHIFT: usize = 2;

impl CGrammar {
    pub fn is_single_byte_terminal(&self, sym: CSymIdx) -> bool {
        sym.0 != 0 && sym.0 <= self.last_single_byte_terminal as u16
    }

    pub fn is_terminal(&self, sym: CSymIdx) -> bool {
        sym.0 != 0 && sym.0 <= self.terminals.len() as u16
    }

    pub fn sym_idx_of(&self, rule: RuleIdx) -> CSymIdx {
        self.rule_idx_to_sym_idx[rule.as_index() >> RULE_SHIFT]
    }

    pub fn sym_flags_of(&self, rule: RuleIdx) -> SymFlags {
        self.rule_idx_to_sym_flags[rule.as_index() >> RULE_SHIFT]
    }

    pub fn rule_rhs(&self, rule: RuleIdx) -> (&[CSymIdx], usize) {
        let idx = rule.as_index();
        let mut start = idx - 1;
        while self.rules[start] != CSymIdx::NULL {
            start -= 1;
        }
        start += 1;
        let mut stop = idx;
        while self.rules[stop] != CSymIdx::NULL {
            stop += 1;
        }
        (&self.rules[start..stop], idx - start)
    }

    pub fn sym_data(&self, sym: CSymIdx) -> &CSymbol {
        &self.symbols[sym.0 as usize]
    }

    pub fn terminal_byteset(&self, sym: CSymIdx) -> &ByteSet {
        &self.terminals[sym.0 as usize]
    }

    fn sym_data_mut(&mut self, sym: CSymIdx) -> &mut CSymbol {
        &mut self.symbols[sym.0 as usize]
    }

    pub fn terminals_by_byte(&self, b: u8) -> &SimpleVob {
        &self.terminals_by_byte[b as usize]
    }

    pub fn sym_idx_at(&self, idx: RuleIdx) -> CSymIdx {
        self.rules[idx.0 as usize]
    }

    #[inline(always)]
    pub fn sym_data_at(&self, idx: RuleIdx) -> &CSymbol {
        self.sym_data(self.sym_idx_at(idx))
    }


    pub fn start(&self) -> CSymIdx {
        self.start_symbol
    }

    pub fn rules_of(&self, sym: CSymIdx) -> &[RuleIdx] {
        &self.sym_data(sym).rules
    }

    fn from_grammar(grammar: &Grammar) -> Self {
        let mut outp = CGrammar {
            start_symbol: CSymIdx::NULL, // replaced
            terminals: vec![ByteSet::new()],
            last_single_byte_terminal: 0,
            symbols: vec![CSymbol {
                idx: CSymIdx::NULL,
                name: "NULL".to_string(),
                is_terminal: true,
                is_nullable: false,
                rules: vec![],
                props: SymbolProps::default(),
                sym_flags: SymFlags(0),
            }],
            rules: vec![CSymIdx::NULL], // make sure RuleIdx::NULL is invalid
            rule_idx_to_sym_idx: vec![],
            terminals_by_byte: vec![],
            rule_idx_to_sym_flags: vec![],
        };
        let mut sym_map = FxHashMap::default();
        let (single, multi) = grammar
            .symbols
            .iter()
            .filter_map(|s| if s.is_terminal() { Some(s) } else { None })
            .partition::<Vec<_>, _>(|s| {
                s.is_lexeme_terminal() && s.bytes.as_ref().unwrap().single_byte().is_some()
            });
        assert!(outp.symbols.len() == 1);
        assert!(outp.terminals.len() == 1);
        // we account for the already existing empty terminal
        outp.last_single_byte_terminal = single.len();
        for sym in single.iter().chain(&multi) {
            outp.terminals
                .push(sym.bytes.clone().unwrap_or_else(ByteSet::new));
            let idx = outp.symbols.len() as u16;
            outp.symbols.push(CSymbol {
                idx: CSymIdx(idx),
                name: sym.name.clone(),
                is_terminal: true,
                is_nullable: false,
                rules: vec![],
                props: sym.props.clone(),
                sym_flags: SymFlags(0),
            });
            sym_map.insert(sym.idx, CSymIdx(idx));
        }
        for sym in &grammar.symbols {
            if sym.is_terminal() {
                continue;
            }
            let idx = outp.symbols.len() as u16;
            outp.symbols.push(CSymbol {
                idx: CSymIdx(idx),
                name: sym.name.clone(),
                is_terminal: false,
                is_nullable: sym.rules.iter().any(|r| r.rhs.is_empty()),
                rules: vec![],
                props: sym.props.clone(),
                sym_flags: SymFlags(0),
            });
            sym_map.insert(sym.idx, CSymIdx(idx));
        }
        outp.start_symbol = sym_map[&grammar.start()];
        for sym in &grammar.symbols {
            if sym.is_terminal() {
                continue;
            }
            let idx = sym_map[&sym.idx];
            for rule in &sym.rules {
                // we handle the empty rule separately via is_nullable field
                if rule.rhs.is_empty() {
                    continue;
                }
                let curr = RuleIdx(outp.rules.len().try_into().unwrap());
                outp.sym_data_mut(idx).rules.push(curr);
                // outp.rules.push(idx);
                for r in &rule.rhs {
                    outp.rules.push(sym_map[r]);
                }
                outp.rules.push(CSymIdx::NULL);
            }
            while outp.rules.len() % (1 << RULE_SHIFT) != 0 {
                outp.rules.push(CSymIdx::NULL);
            }
            let rlen = outp.rules.len() >> RULE_SHIFT;
            while outp.rule_idx_to_sym_idx.len() < rlen {
                outp.rule_idx_to_sym_idx.push(idx);
            }
        }

        for sym in &mut outp.symbols {
            sym.sym_flags = SymFlags::from_csymbol(sym);
        }

        outp.rule_idx_to_sym_flags = outp
            .rule_idx_to_sym_idx
            .iter()
            .map(|s| outp.sym_data(*s).sym_flags)
            .collect();

        loop {
            let mut to_null = vec![];
            for sym in &outp.symbols {
                if sym.is_nullable {
                    continue;
                }
                for rule in sym.rules.iter() {
                    if outp
                        .rule_rhs(*rule)
                        .0
                        .iter()
                        .all(|elt| outp.sym_data(*elt).is_nullable)
                    {
                        to_null.push(sym.idx);
                    }
                }
            }
            if to_null.is_empty() {
                break;
            }
            for sym in to_null {
                outp.sym_data_mut(sym).is_nullable = true;
            }
        }

        for b in 0..=255 {
            let mut v = SimpleVob::alloc(outp.terminals.len());
            for (i, bytes) in outp.terminals.iter().enumerate() {
                if bytes.contains(b as u8) {
                    v.allow_token(i as u32);
                }
            }
            outp.terminals_by_byte.push(v);
        }
        outp
    }

    pub fn sym_name(&self, sym: CSymIdx) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    pub fn rule_to_string(&self, rule: RuleIdx) -> String {
        let sym = self.sym_idx_of(rule);
        let symdata = self.sym_data(sym);
        let lhs = self.sym_name(sym);
        let (rhs, dot) = self.rule_rhs(rule);
        let dot_prop = if rhs.len() > 0 {
            Some(&self.sym_data_at(rule).props)
        } else {
            None
        };
        rule_to_string(
            lhs,
            rhs.iter()
                .map(|s| {
                    let d = self.sym_data(*s);
                    SymName::from(
                        &d.name,
                        if d.is_terminal {
                            Some(&self.terminals[d.idx.0 as usize])
                        } else {
                            None
                        },
                    )
                })
                .collect(),
            Some(dot),
            &symdata.props,
            dot_prop,
        )
    }
}

fn rule_to_string(
    lhs: &str,
    mut rhs: Vec<SymName>,
    dot: Option<usize>,
    props: &SymbolProps,
    _dot_props: Option<&SymbolProps>,
) -> String {
    if rhs.is_empty() {
        rhs.push(SymName::Name("ϵ".to_string()));
        if dot == Some(0) {
            rhs.push(SymName::Name("•".to_string()));
        }
    } else if let Some(dot) = dot {
        rhs.insert(dot, SymName::Name("•".to_string()));
    }
    let mut outp = Vec::new();
    let mut i = 0;
    while i < rhs.len() {
        match &rhs[i] {
            SymName::Name(s) => {
                outp.push(s.clone());
                i += 1;
            }
            SymName::Byte(_) => {
                let mut text = Vec::new();
                while i < rhs.len() {
                    if let SymName::Byte(b) = rhs[i] {
                        text.push(b);
                        i += 1;
                    } else {
                        break;
                    }
                }
                outp.push(format!("{:?}", String::from_utf8_lossy(&text)));
            }
        }
    }
    format!(
        "{:15} ⇦ {}  {}{}{}",
        lhs,
        outp.join(" "),
        if props.commit_point {
            if props.hidden {
                " HIDDEN-COMMIT"
            } else {
                " COMMIT"
            }
        } else {
            ""
        },
        if props.capture_name.is_some() {
            " CAPTURE"
        } else {
            ""
        },
        if props.max_tokens < 1000 {
            format!(" max_tokens={}", props.max_tokens)
        } else {
            "".to_string()
        },
    )
}
