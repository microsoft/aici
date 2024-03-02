use std::{fmt::Debug, vec};

use super::{
    byteset::byte_to_string,
    grammar::{OptGrammar, OptSymIdx, RuleIdx},
};

const DEBUG: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    rule_idx: RuleIdx,
    start: u32,
    sym_idx: OptSymIdx,
}

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
    fn new(sym: OptSymIdx, rule: RuleIdx, start: usize) -> Self {
        Item {
            sym_idx: sym,
            rule_idx: rule,
            start: start.try_into().unwrap(),
        }
    }

    fn rule_idx(&self) -> RuleIdx {
        self.rule_idx
    }

    fn sym_idx(&self) -> OptSymIdx {
        self.sym_idx
    }

    fn start_pos(&self) -> usize {
        self.start as usize
    }

    fn advance_dot(&self) -> Self {
        Item::new(self.sym_idx, self.rule_idx.advance(), self.start_pos())
    }
}

pub struct Parser {
    grammar: OptGrammar,
    rows: Vec<Row>,
}

impl Parser {
    pub fn new(grammar: OptGrammar) -> Self {
        let start = grammar.start();
        let init_rules = grammar
            .rules_of(start)
            .iter()
            .map(|r| Item::new(start, *r, 0))
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
        // let rule = self.grammar.rule_data(item.rule_idx());
        // self.grammar.rule_to_string(rule, item.dot_pos())
        format!(
            "item: rule: {:?}, dot: {:?}, start: {}",
            item.rule_idx, item.sym_idx, item.start
        )
    }

    pub fn row_to_string(&self, row: &Row) -> String {
        let mut r = vec![format!("token: {}", byte_to_string(row.token))];
        for item in &row.items {
            r.push(self.item_to_string(item));
        }
        r.join("\n") + "\n"
    }

    pub fn scan(&self, b: u8) -> Row {
        let allowed = self.grammar.terminals_by_byte(b);
        let mut r = vec![];
        let row_idx = self.rows.len() - 1;
        for item in &self.rows[row_idx].items {
            let idx = self.grammar.sym_idx_at(item.rule_idx()).as_index();
            assert!(idx != 0);
            if idx < allowed.len() && allowed[idx] {
                r.push(item.advance_dot());
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
            let mut to_add = vec![];
            let mut add = |new_item: Item, tag: &str| {
                if !to_add.contains(&new_item) {
                    to_add.push(new_item);
                    if DEBUG {
                        println!("  adding {}: {}", tag, self.item_to_string(&new_item));
                    }
                }
            };

            let lhs = item.sym_idx();
            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_at(rule);

            if after_dot == OptSymIdx::NULL {
                // complete
                if lhs == self.grammar.start() {
                    accepting = true;
                }

                if item.start_pos() < curr_idx {
                    // if item.start_pos() == curr_idx, then we handled it above in the nullable check
                    for item in self.rows[item.start_pos()].items.iter() {
                        if self.grammar.sym_idx_at(item.rule_idx()) == lhs {
                            add(item.advance_dot(), "complete");
                        }
                    }
                }
            } else {
                let sym_data = self.grammar.sym_data(after_dot);
                if sym_data.is_nullable {
                    add(item.advance_dot(), "null");
                }
                if !predicated_syms.contains(&after_dot) {
                    predicated_syms.push(after_dot);
                    for rule in &sym_data.rules {
                        let new_item = Item::new(after_dot, *rule, curr_idx);
                        add(new_item, "predict");
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
