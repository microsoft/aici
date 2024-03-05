use std::fmt::Debug;

use crate::svob::SimpleVob;

use super::ByteSet;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymIdx(u32);

impl Symbol {
    fn is_terminal(&self) -> bool {
        self.bytes.is_some()
    }
}

struct Symbol {
    idx: SymIdx,
    name: String,
    bytes: Option<ByteSet>,
    rules: Vec<Rule>,
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

pub struct Grammar {
    symbols: Vec<Symbol>,
    symbol_by_name: FxHashMap<String, SymIdx>,
    terminals: FxHashMap<ByteSet, SymIdx>,
}

impl Grammar {
    pub fn new() -> Self {
        let mut r = Grammar {
            symbols: vec![],
            symbol_by_name: FxHashMap::default(),
            terminals: FxHashMap::default(),
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

    pub fn terminal(&mut self, bytes: ByteSet) -> SymIdx {
        match self.terminals.get(&bytes) {
            Some(sym) => *sym,
            None => {
                let mut name = format!("T:{}", bytes);
                if name.len() > 40 {
                    name = format!("T@{}", self.terminals.len());
                }
                let sym = self.fresh_symbol(&name);
                self.sym_data_mut(sym).bytes = Some(bytes.clone());
                self.terminals.insert(bytes, sym);
                sym
            }
        }
    }

    pub fn sym_name(&self, sym: SymIdx) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    fn rule_to_string(&self, rule: &Rule, dot: usize) -> String {
        let lhs = self.sym_name(rule.lhs());
        let mut rhs = rule
            .rhs
            .iter()
            .enumerate()
            .map(|(i, s)| {
                format!(
                    "{}{}",
                    if i == dot { "(*) " } else { "" },
                    self.sym_name(*s)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        if rule.rhs.is_empty() {
            rhs.push_str("ϵ");
        }
        if dot == rule.rhs.len() {
            rhs.push_str(" (*)");
        }
        format!("{} ::= {}", lhs, rhs)
    }

    fn copy_from(&mut self, other: &Grammar, sym: SymIdx) -> SymIdx {
        let sym_data = other.sym_data(sym);
        if sym_data.is_terminal() {
            self.terminal(sym_data.bytes.clone().unwrap())
        } else {
            self.symbol(&sym_data.name)
        }
    }

    fn collapse_terminals(&self) -> Self {
        let mut outp = Grammar::new();
        for sym in &self.symbols {
            if sym.rules.is_empty() {
                continue;
            }
            let mut rules_by_shape = FxHashMap::default();
            for rule in &sym.rules {
                let shape = rule
                    .rhs
                    .iter()
                    .map(|s| {
                        if self.sym_data(*s).is_terminal() {
                            None
                        } else {
                            Some(*s)
                        }
                    })
                    .collect::<Vec<_>>();
                rules_by_shape
                    .entry(shape)
                    .or_insert_with(Vec::new)
                    .push(rule);
            }
            let lhs = outp.copy_from(self, sym.idx);
            for rules in rules_by_shape.values() {
                let rhs = rules[0]
                    .rhs
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let sym = self.sym_data(*s);
                        if sym.is_terminal() {
                            let terminals = rules
                                .iter()
                                .map(|r| self.sym_data(r.rhs[i]).bytes.clone().unwrap());
                            outp.terminal(ByteSet::from_sum(terminals))
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
            if sym.idx == self.start() {
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
        self.expand_shortcuts()
            .collapse_terminals()
            .expand_shortcuts()
    }

    pub fn compile(&self) -> OptGrammar {
        OptGrammar::from_grammar(self)
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
                writeln!(f, "{}", self.rule_to_string(rule, usize::MAX))?;
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
pub struct OptSymIdx(u16);

impl OptSymIdx {
    pub const NULL: OptSymIdx = OptSymIdx(0);

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

impl SimpleHash for OptSymIdx {
    fn simple_hash(&self) -> u32 {
        (self.0 as u32).wrapping_mul(79667123)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleIdx(u32);

impl RuleIdx {
    pub fn advance(&self) -> RuleIdx {
        RuleIdx(self.0 + 1)
    }

    pub fn as_index(&self) -> usize {
        self.0 as usize
    }
}

pub struct OptSymbol {
    pub idx: OptSymIdx,
    pub name: String,
    pub is_terminal: bool,
    pub is_nullable: bool,
    pub rules: Vec<RuleIdx>,
}

pub struct OptGrammar {
    start_symbol: OptSymIdx,
    terminals: Vec<ByteSet>,
    symbols: Vec<OptSymbol>,
    rules: Vec<OptSymIdx>,
    rule_idx_to_sym_idx: Vec<OptSymIdx>,
    terminals_by_byte: Vec<SimpleVob>,
}

const RULE_SHIFT: usize = 2;

impl OptGrammar {
    pub fn sym_idx_of(&self, rule: RuleIdx) -> OptSymIdx {
        self.rule_idx_to_sym_idx[rule.as_index() >> RULE_SHIFT]
    }

    pub fn sym_data(&self, sym: OptSymIdx) -> &OptSymbol {
        &self.symbols[sym.0 as usize]
    }

    fn sym_data_mut(&mut self, sym: OptSymIdx) -> &mut OptSymbol {
        &mut self.symbols[sym.0 as usize]
    }

    pub fn terminals_by_byte(&self, b: u8) -> &SimpleVob {
        &self.terminals_by_byte[b as usize]
    }

    pub fn sym_idx_at(&self, idx: RuleIdx) -> OptSymIdx {
        self.rules[idx.0 as usize]
    }

    pub fn start(&self) -> OptSymIdx {
        self.start_symbol
    }

    pub fn is_accepting(&self, sym: OptSymIdx, rule: RuleIdx) -> bool {
        sym == self.start() && self.sym_idx_at(rule) == OptSymIdx::NULL
    }

    pub fn rules_of(&self, sym: OptSymIdx) -> &[RuleIdx] {
        &self.sym_data(sym).rules
    }

    fn from_grammar(grammar: &Grammar) -> Self {
        let mut outp = OptGrammar {
            start_symbol: OptSymIdx::NULL,
            terminals: vec![ByteSet::new()],
            symbols: vec![OptSymbol {
                idx: OptSymIdx::NULL,
                name: "NULL".to_string(),
                is_terminal: true,
                is_nullable: false,
                rules: vec![],
            }],
            rules: vec![],
            rule_idx_to_sym_idx: vec![],
            terminals_by_byte: vec![],
        };
        let mut sym_map = FxHashMap::default();
        for (_, sidx) in &grammar.terminals {
            let sym = grammar.sym_data(*sidx);
            outp.terminals.push(sym.bytes.clone().unwrap());
            let idx = outp.symbols.len() as u16;
            outp.symbols.push(OptSymbol {
                idx: OptSymIdx(idx),
                name: sym.name.clone(),
                is_terminal: true,
                is_nullable: false,
                rules: vec![],
            });
            sym_map.insert(sym.idx, OptSymIdx(idx));
        }
        for sym in &grammar.symbols {
            if sym.is_terminal() {
                continue;
            }
            let idx = outp.symbols.len() as u16;
            outp.symbols.push(OptSymbol {
                idx: OptSymIdx(idx),
                name: sym.name.clone(),
                is_terminal: false,
                is_nullable: sym.rules.iter().any(|r| r.rhs.is_empty()),
                rules: vec![],
            });
            sym_map.insert(sym.idx, OptSymIdx(idx));
        }
        outp.start_symbol = sym_map[&grammar.start()];
        for sym in &grammar.symbols {
            if sym.is_terminal() {
                continue;
            }
            let idx = sym_map[&sym.idx];
            for rule in &sym.rules {
                let curr = RuleIdx(outp.rules.len().try_into().unwrap());
                outp.sym_data_mut(idx).rules.push(curr);
                // outp.rules.push(idx);
                for r in &rule.rhs {
                    outp.rules.push(sym_map[r]);
                }
                outp.rules.push(OptSymIdx::NULL);
            }
            while outp.rules.len() % (1 << RULE_SHIFT) != 0 {
                outp.rules.push(OptSymIdx::NULL);
            }
            let rlen = outp.rules.len() >> RULE_SHIFT;
            while outp.rule_idx_to_sym_idx.len() < rlen {
                outp.rule_idx_to_sym_idx.push(idx);
            }
        }

        loop {
            let mut to_null = vec![];
            for sym in &outp.symbols {
                if sym.is_nullable {
                    continue;
                }
                'rules: for rule in sym.rules.iter() {
                    let mut idx = rule.as_index();
                    while outp.rules[idx] != OptSymIdx::NULL {
                        if outp.sym_data(outp.rules[idx]).is_nullable {
                            to_null.push(sym.idx);
                            break 'rules;
                        }
                        idx += 1;
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
}
