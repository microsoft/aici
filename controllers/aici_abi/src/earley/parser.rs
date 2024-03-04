use std::{fmt::Debug, hash::Hash, ops::Range, vec};

use super::grammar::{OptGrammar, OptSymIdx, RuleIdx, SimpleHash};

const DEBUG: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    rule_idx: RuleIdx,
    start: u32,
    sym_idx: OptSymIdx,
}

#[derive(Debug, Default)]
pub struct Stats {
    pub rows: usize,
    pub empty_rows: usize,
    pub nontrivial_scans: usize,
    pub scan_items: usize,
    pub all_items: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseResult {
    Accept,
    Reject,
    Continue,
}

struct Row {
    first_item: usize,
    last_item: usize,
}

impl Row {
    fn item_indices(&self) -> Range<usize> {
        self.first_item..self.last_item
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

impl SimpleHash for Item {
    fn simple_hash(&self) -> u32 {
        (self.rule_idx.as_index() as u32)
            .wrapping_mul(16315967)
            .wrapping_add((self.start as u32).wrapping_mul(33398653))
    }
}

struct SimpleSet<T: SimpleHash + Eq> {
    hash: u64,
    items: Vec<T>,
}

impl<T: SimpleHash + Eq + Copy> Default for SimpleSet<T> {
    fn default() -> Self {
        SimpleSet {
            hash: 0,
            items: vec![],
        }
    }
}

impl<T: SimpleHash + Eq + Copy> SimpleSet<T> {
    fn clear(&mut self) {
        self.hash = 0;
        self.items.clear();
    }

    fn insert(&mut self, item: T) {
        let mask = item.mask64();
        if (self.hash & mask) != 0 && self.items.contains(&item) {
            return;
        }
        self.hash |= mask;
        self.items.push(item);
    }

    fn contains(&self, item: T) -> bool {
        if (item.mask64() & self.hash) == 0 {
            false
        } else {
            self.items.contains(&item)
        }
    }
}

#[derive(Default)]
struct Scratch {
    row_start: usize,
    row_end: usize,
    items: Vec<Item>,
    predicated_syms: SimpleSet<OptSymIdx>,
}

pub struct Parser {
    grammar: OptGrammar,
    scratch: Scratch,
    rows: Vec<Row>,
    stats: Stats,
    is_accepting: bool,
}

impl Scratch {
    fn new_row(&mut self, pos: usize) {
        self.row_start = pos;
        self.row_end = pos;
    }

    fn row_len(&self) -> usize {
        self.row_end - self.row_start
    }

    #[inline(always)]
    fn ensure_items(&mut self, n: usize) {
        if self.items.len() < n {
            let missing = n - self.items.len();
            self.items.reserve(missing);
            unsafe { self.items.set_len(n) }
        }
    }

    #[inline(always)]
    fn just_add(&mut self, item: Item) {
        self.ensure_items(self.row_end + 1);
        self.items[self.row_end] = item;
        self.row_end += 1;
    }

    #[inline(always)]
    fn add_unique(&mut self, item: Item, _info: &str) {
        if !self.items[self.row_start..self.row_end].contains(&item) {
            self.just_add(item);
        }
    }
}

impl Parser {
    pub fn new(grammar: OptGrammar) -> Self {
        let start = grammar.start();
        let mut r = Parser {
            grammar,
            rows: vec![],
            scratch: Scratch::default(),
            stats: Stats::default(),
            is_accepting: false,
        };
        for rule in r.grammar.rules_of(start).to_vec() {
            r.scratch.add_unique(Item::new(start, rule, 0), "init");
        }
        let _ = r.push_row();
        r
    }

    pub fn is_accepting(&self) -> bool {
        self.is_accepting
    }

    fn item_to_string(&self, item: &Item) -> String {
        // let rule = self.grammar.rule_data(item.rule_idx());
        // self.grammar.rule_to_string(rule, item.dot_pos())
        format!(
            "item: rule: {:?}, dot: {:?}, start: {}",
            item.rule_idx, item.sym_idx, item.start
        )
    }

    // fn row_to_string(&self, row: &Row) -> String {
    //     // let mut r = vec![format!("token: {}", byte_to_string(row.token))];
    //     // for item in &row.items {
    //     //     r.push(self.item_to_string(item));
    //     // }
    //     // r.join("\n") + "\n"
    //     "todo".to_string()
    // }

    #[inline(always)]
    pub fn scan(&mut self, b: u8) -> ParseResult {
        let row_idx = self.rows.len() - 1;
        let last = self.rows[row_idx].last_item;
        let mut i = self.rows[row_idx].first_item;
        let n = last - i;
        self.scratch.ensure_items(last + n + 100);

        let allowed = self.grammar.terminals_by_byte(b);

        self.scratch.new_row(last);

        while i < last {
            let item = self.scratch.items[i];
            let idx = self.grammar.sym_idx_at(item.rule_idx()).as_index();
            // idx == 0 => completed
            if idx < allowed.len() && allowed[idx] {
                self.scratch.just_add(item.advance_dot());
            }
            i += 1;
        }
        self.push_row()
    }

    pub fn pop_rows(&mut self, n: usize) {
        unsafe { self.rows.set_len(self.rows.len() - n) }
        // self.rows.drain(self.rows.len() - n..);
    }

    pub fn print_stats(&mut self) {
        println!("stats: {:?}", self.stats);
        self.stats = Stats::default();
    }

    #[inline(always)]
    fn push_row(&mut self) -> ParseResult {
        let curr_idx = self.rows.len();
        let mut agenda_ptr = self.scratch.row_start;

        self.scratch.predicated_syms.clear();

        self.stats.rows += 1;
        self.is_accepting = false;

        while agenda_ptr < self.scratch.row_end {
            let item = self.scratch.items[agenda_ptr];
            agenda_ptr += 1;
            if DEBUG {
                println!("from agenda: {}", self.item_to_string(&item));
            }

            let lhs = item.sym_idx();
            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_at(rule);

            if after_dot == OptSymIdx::NULL {
                // complete
                self.is_accepting = self.is_accepting || lhs == self.grammar.start();

                if item.start_pos() < curr_idx {
                    // if item.start_pos() == curr_idx, then we handled it above in the nullable check
                    for i in self.rows[item.start_pos()].item_indices() {
                        let item = self.scratch.items[i];
                        if self.grammar.sym_idx_at(item.rule_idx()) == lhs {
                            self.scratch.add_unique(item.advance_dot(), "complete");
                        }
                    }
                }
            } else {
                let sym_data = self.grammar.sym_data(after_dot);
                if sym_data.is_nullable {
                    self.scratch.add_unique(item.advance_dot(), "null");
                }
                if !self.scratch.predicated_syms.contains(after_dot) {
                    self.scratch.predicated_syms.insert(after_dot);
                    for rule in &sym_data.rules {
                        let new_item = Item::new(after_dot, *rule, curr_idx);
                        self.scratch.add_unique(new_item, "predict");
                    }
                }
            }
        }

        let row_len = self.scratch.row_len();
        self.stats.all_items += row_len;

        if row_len == 0 {
            assert!(!self.is_accepting);
            return ParseResult::Reject;
        }

        self.rows.push(Row {
            first_item: self.scratch.row_start,
            last_item: self.scratch.row_end,
        });

        if self.is_accepting {
            ParseResult::Accept
        } else {
            ParseResult::Continue
        }
    }
}
