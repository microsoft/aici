use std::{fmt::Debug, rc::Rc, vec};

use rustc_hash::FxHashMap;

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

impl RuleIdx {
    fn sym_idx(&self) -> SymIdx {
        SymIdx(self.data >> RULE_IDX_BITS)
    }

    fn sym_rule_idx(&self) -> usize {
        (self.data & mask32(RULE_IDX_BITS)) as usize
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
    rx: Option<String>,
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
}

// format:
// token_position : dot_position : symbol_index : rule_index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item(u64);

pub struct Row {
    token: SymIdx,
    // TODO index this by .after_dot() ?
    items: Vec<Item>,
}

impl Row {

}

impl Item {
    fn new(rule: RuleIdx, dot: usize, start: usize) -> Self {
        assert!(start < mask32(TOK_POS_BITS) as usize);
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
        (self.0 >> (DOT_POS_BITS + SYM_IDX_BITS + RULE_IDX_BITS)) as usize
            & mask32(TOK_POS_BITS) as usize
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
    pub fn new(grammar: Rc<Grammar>) -> Self {
        let start = grammar.start();
        let init_rules = grammar
            .sym_data(start)
            .rules
            .iter()
            .map(|r| Item::new(r.idx, 0, 0))
            .collect();
        let mut r = Parser {
            grammar,
            rows: vec![],
        };
        // 'start' token is bogus
        r.push_row(init_rules, start);
        r
    }

    fn after_dot(&self, item: Item) -> Option<SymIdx> {
        let rule = self.grammar.rule_data(item.rule_idx());
        if item.dot_pos() < rule.rhs.len() {
            Some(rule.rhs[item.dot_pos()])
        } else {
            None
        }
    }

    fn scan(&mut self, token: SymIdx) {
        let next_row = self.items_with_after_dot(token, self.rows.len() - 1);
        self.push_row(next_row, token);
    }

    fn items_with_after_dot(&self, sym: SymIdx, row_idx: usize) -> Vec<Item> {
        let mut r = vec![];
        for item in &self.rows[row_idx].items {
            if self.after_dot(*item) == Some(sym) {
                r.push(*item);
            }
        }
        r
    }

    fn push_row(&mut self, mut curr_row: Vec<Item>, token: SymIdx) {
        let curr_idx = self.rows.len();
        let mut agenda = curr_row.clone();
        let mut predicated_syms = vec![];

        while !agenda.is_empty() {
            let item = agenda.pop().unwrap();
            let lhs = item.rule_idx().sym_idx();
            let mut to_add = vec![];
            match self.after_dot(item) {
                Some(after_dot) => {
                    let sym_data = self.grammar.sym_data(after_dot);
                    if sym_data.nullable {
                        let new_item = item.advance_dot();
                        if !to_add.contains(&new_item) {
                            to_add.push(new_item);
                        }
                    }
                    if !predicated_syms.contains(&after_dot) {
                        predicated_syms.push(after_dot);
                        for rule in &sym_data.rules {
                            let new_item = Item::new(rule.idx, 0, curr_idx);
                            if !to_add.contains(&new_item) {
                                to_add.push(new_item);
                            }
                        }
                    }
                }
                // complete
                None => {
                    if item.start_pos() < curr_idx {
                        // if item.start_pos() == curr_idx, then we handled it above in the nullable check
                        for parent in self.items_with_after_dot(lhs, item.start_pos()) {
                            let new_item = parent.advance_dot();
                            if !to_add.contains(&new_item) {
                                to_add.push(new_item);
                            }
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

        self.rows.push(Row {
            token,
            items: curr_row,
        });
    }
}

impl Grammar {
    pub fn new() -> Self {
        let mut r = Grammar {
            symbols: vec![],
            symbol_by_name: FxHashMap::default(),
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
        &sym.rules[rule.sym_rule_idx()]
    }

    fn propagate_nullable(&mut self) {
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

        let is_nullable = rhs.iter().all(|s| self.sym_data(*s).nullable);

        if rhs.len() > 0 {
            let sym = self.sym_data_mut(lhs);
            sym.rules.push(Rule {
                idx: lhs.rule_at(sym.rules.len()),
                rhs,
            });
        }

        if is_nullable {
            self.sym_data_mut(lhs).nullable = true;
            self.propagate_nullable();
        }
    }

    pub fn make_terminal(&mut self, sym: SymIdx, rx: &str) {
        self.symbols[sym.0 as usize].rx = Some(rx.to_string());
    }

    pub fn sym_name(&self, sym: SymIdx) -> &str {
        &self.symbols[sym.0 as usize].name
    }

    fn rule_to_string(&self, rule: &Rule) -> String {
        let lhs = self.sym_name(rule.lhs());
        let rhs = rule
            .rhs
            .iter()
            .map(|s| self.sym_name(*s))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} ::= {}", lhs, rhs)
    }

    pub fn symbol(&mut self, name: &str) -> SymIdx {
        match self.symbol_by_name.get(name) {
            Some(idx) => *idx,
            None => {
                let idx = SymIdx(self.symbols.len() as u32);
                self.symbols.push(Symbol {
                    name: name.to_string(),
                    rx: None,
                    idx,
                    rules: vec![],
                    nullable: false,
                });
                self.symbol_by_name.insert(name.to_string(), idx);
                idx
            }
        }
    }
}

impl Debug for Grammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for sym in &self.symbols {
            match sym.rx {
                Some(ref rx) => writeln!(f, "{} /= {:?}", sym.name, rx)?,
                None => {}
            }
        }
        for sym in &self.symbols {
            for rule in &sym.rules {
                writeln!(f, "{}", self.rule_to_string(rule))?;
            }
        }
        Ok(())
    }
}
