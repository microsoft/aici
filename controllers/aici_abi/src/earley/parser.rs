use std::{fmt::Debug, rc::Rc, vec};

use rustc_hash::FxHashMap;

use super::{byteset::byte_to_string, ByteSet};

const DEBUG: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymIdx(u32);

// format:
// symbol_index : rule_index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleIdx {
    data: u32,
}

const SYM_IDX_BITS: u32 = 12;
const RULE_IDX_BITS: u32 = 10;
const DOT_POS_BITS: u32 = 7;
const TOK_POS_BITS: u32 = 64 - (DOT_POS_BITS + SYM_IDX_BITS + RULE_IDX_BITS);

fn mask32(bits: u32) -> u32 {
    (1 << bits) - 1
}

fn mask64(bits: u32) -> u64 {
    (1u64 << (bits as u64)) - 1
}

impl RuleIdx {
    fn sym_idx(&self) -> SymIdx {
        SymIdx(self.data >> RULE_IDX_BITS)
    }

    fn sym_rule_idx(&self) -> usize {
        (self.data & mask32(RULE_IDX_BITS)) as usize
    }
}

impl Symbol {
    fn is_terminal(&self) -> bool {
        self.bytes.is_some()
    }
}

impl SymIdx {
    fn rule_at(&self, rule: usize) -> RuleIdx {
        assert!(rule < mask32(RULE_IDX_BITS) as usize);
        RuleIdx {
            data: (self.0 << RULE_IDX_BITS) | rule as u32,
        }
    }
}

struct Symbol {
    idx: SymIdx,
    name: String,
    bytes: Option<ByteSet>,
    rules: Vec<Rule>,
    nullable: bool,
}

struct Rule {
    idx: RuleIdx,
    rhs: Vec<SymIdx>,
}

impl Rule {
    fn lhs(&self) -> SymIdx {
        self.idx.sym_idx()
    }
}

pub struct Grammar {
    symbols: Vec<Symbol>,
    symbol_by_name: FxHashMap<String, SymIdx>,
    terminals: FxHashMap<ByteSet, SymIdx>,
}

#[derive(Clone)]
pub struct OptimizedGrammar {
    inner: Rc<Grammar>,
}

impl Debug for OptimizedGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

// format:
// token_position : dot_position : symbol_index : rule_index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item(u64);

pub struct Row {
    token: u8,
    position: usize,
    // TODO index this by .after_dot() ?
    items: Vec<Item>,
    accepting: bool,
}

impl Row {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn is_accepting(&self) -> bool {
        self.accepting
    }
}

impl Item {
    fn new(rule: RuleIdx, dot: usize, start: usize) -> Self {
        assert!(start < mask64(TOK_POS_BITS) as usize);
        assert!(dot < mask32(DOT_POS_BITS) as usize);
        let data = (start as u64) << (DOT_POS_BITS + SYM_IDX_BITS + RULE_IDX_BITS)
            | (dot as u64) << (SYM_IDX_BITS + RULE_IDX_BITS)
            | (rule.data as u64);
        Item(data)
    }

    fn rule_idx(&self) -> RuleIdx {
        RuleIdx {
            data: self.0 as u32 & mask32(SYM_IDX_BITS + RULE_IDX_BITS),
        }
    }

    fn dot_pos(&self) -> usize {
        (self.0 >> (SYM_IDX_BITS + RULE_IDX_BITS)) as usize & mask32(DOT_POS_BITS) as usize
    }

    fn start_pos(&self) -> usize {
        ((self.0 >> (DOT_POS_BITS + SYM_IDX_BITS + RULE_IDX_BITS)) & mask64(TOK_POS_BITS)) as usize
    }

    fn advance_dot(&self) -> Self {
        Item::new(self.rule_idx(), self.dot_pos() + 1, self.start_pos())
    }
}

pub struct Parser {
    grammar: Rc<Grammar>,
    rows: Vec<Row>,
}

impl Parser {
    pub fn new(grammar: OptimizedGrammar) -> Self {
        let grammar = grammar.inner;
        let init_rules = grammar
            .sym_data(grammar.start())
            .rules
            .iter()
            .map(|r| Item::new(r.idx, 0, 0))
            .collect();
        let mut r = Parser {
            grammar,
            rows: vec![],
        };
        // '0' token is bogus
        let row = r.make_row(init_rules, 0);
        println!("init: {}", r.row_to_string(&row));
        r.push_row(row);
        r
    }

    fn item_to_string(&self, item: &Item) -> String {
        let rule = self.grammar.rule_data(item.rule_idx());
        self.grammar.rule_to_string(rule, item.dot_pos())
    }

    pub fn row_to_string(&self, row: &Row) -> String {
        let mut r = vec![format!("token: {}", byte_to_string(row.token))];
        for item in &row.items {
            r.push(self.item_to_string(item));
        }
        r.join("\n") + "\n"
    }

    pub fn scan(&mut self, b: u8) -> Row {
        let mut r = vec![];
        let row_idx = self.rows.len() - 1;
        for item in &self.rows[row_idx].items {
            if let Some(s) = self.grammar.after_dot(*item) {
                if let Some(bytes) = self.grammar.sym_data(s).bytes.clone() {
                    if bytes.contains(b) {
                        r.push(item.advance_dot());
                    }
                }
            }
        }
        self.make_row(r, b)
    }

    pub fn pop_rows(&mut self, n: usize) {
        self.rows.drain(self.rows.len() - n..);
    }

    pub fn curr_row(&self) -> &Row {
        &self.rows[self.rows.len() - 1]
    }

    pub fn push_row(&mut self, row: Row) {
        assert!(row.position == self.rows.len());
        self.rows.push(row);
    }

    fn items_with_after_dot(&self, sym: SymIdx, row_idx: usize) -> Vec<Item> {
        let mut r = vec![];
        for item in &self.rows[row_idx].items {
            if self.grammar.after_dot(*item) == Some(sym) {
                r.push(*item);
            }
        }
        r
    }

    fn make_row(&self, mut curr_row: Vec<Item>, token: u8) -> Row {
        let curr_idx = self.rows.len();
        let mut agenda = curr_row.clone();
        let mut predicated_syms = vec![];
        let mut accepting = false;

        if DEBUG {
            let row0 = Row {
                token,
                position: curr_idx,
                items: curr_row.clone(),
                accepting,
            };
            println!("row0: {}", self.row_to_string(&row0));
        }

        while !agenda.is_empty() {
            let item = agenda.pop().unwrap();
            if DEBUG {
                println!("from agenda: {}", self.item_to_string(&item));
            }
            let lhs = item.rule_idx().sym_idx();
            if lhs == self.grammar.start() && self.grammar.after_dot(item).is_none() {
                accepting = true;
            }
            let mut to_add = vec![];
            let mut add = |new_item: Item, tag: &str| {
                if !to_add.contains(&new_item) {
                    to_add.push(new_item);
                    if DEBUG {
                        println!("  adding {}: {}", tag, self.item_to_string(&new_item));
                    }
                }
            };
            match self.grammar.after_dot(item) {
                Some(after_dot) => {
                    let sym_data = self.grammar.sym_data(after_dot);
                    if sym_data.nullable {
                        let new_item = item.advance_dot();
                        add(new_item, "null");
                    }
                    if !predicated_syms.contains(&after_dot) {
                        predicated_syms.push(after_dot);
                        for rule in &sym_data.rules {
                            let new_item = Item::new(rule.idx, 0, curr_idx);
                            add(new_item, "predict");
                        }
                    }
                }
                // complete
                None => {
                    if item.start_pos() < curr_idx {
                        // if item.start_pos() == curr_idx, then we handled it above in the nullable check
                        for parent in self.items_with_after_dot(lhs, item.start_pos()) {
                            let new_item = parent.advance_dot();
                            add(new_item, "complete");
                        }
                    }
                }
            }

            for new_item in to_add {
                if !curr_row.contains(&new_item) {
                    curr_row.push(new_item);
                    agenda.push(new_item);
                }
            }
        }

        Row {
            token,
            position: curr_idx,
            items: curr_row,
            accepting,
        }
    }
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

    fn rule_data(&self, rule: RuleIdx) -> &Rule {
        let sym = self.sym_data(rule.sym_idx());
        if rule.sym_rule_idx() >= sym.rules.len() {
            println!("invalid rule index; {}", sym.name);
        }
        &sym.rules[rule.sym_rule_idx()]
    }

    fn propagate_nullable(&mut self) {
        for sym in self.symbols.iter_mut() {
            if sym.rules.iter().any(|r| r.rhs.is_empty()) {
                sym.nullable = true;
                sym.rules.retain(|r| !r.rhs.is_empty());
                // re-number them
                for (i, r) in sym.rules.iter_mut().enumerate() {
                    r.idx = sym.idx.rule_at(i);
                }
            }
        }
        loop {
            let mut to_null = vec![];
            for sym in self.symbols.iter() {
                for rule in sym.rules.iter() {
                    if rule.rhs.iter().all(|s| self.sym_data(*s).nullable) {
                        if !sym.nullable {
                            to_null.push(sym.idx);
                        }
                    }
                }
            }
            if to_null.is_empty() {
                break;
            }
            for sym in to_null {
                self.sym_data_mut(sym).nullable = true;
            }
        }
    }

    pub fn add_rule(&mut self, lhs: SymIdx, rhs: Vec<SymIdx>) {
        assert!(rhs.len() < mask32(DOT_POS_BITS) as usize);
        assert!(!self.sym_data(lhs).is_terminal());
        let sym = self.sym_data_mut(lhs);
        sym.rules.push(Rule {
            idx: lhs.rule_at(sym.rules.len()),
            rhs,
        });
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

    pub fn optimize(&self) -> OptimizedGrammar {
        let mut outp = self
            .expand_shortcuts()
            .collapse_terminals()
            .expand_shortcuts();
        outp.propagate_nullable();
        OptimizedGrammar {
            inner: Rc::new(outp),
        }
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
            nullable: false,
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

    fn after_dot(&self, item: Item) -> Option<SymIdx> {
        let rule = self.rule_data(item.rule_idx());
        if item.dot_pos() < rule.rhs.len() {
            Some(rule.rhs[item.dot_pos()])
        } else {
            None
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
            if sym.nullable {
                writeln!(f, "{} ::= ϵ", sym.name)?;
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
