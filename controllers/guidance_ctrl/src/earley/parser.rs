use std::{fmt::Debug, hash::Hash, ops::Range, vec};

use aici_abi::{
    toktree::{Recognizer, SpecialToken, TokTrie},
    TokenId,
};

use super::grammar::{CGrammar, CSymIdx, CSymbol, ModelVariable, RuleIdx, SimpleHash};

const DEBUG: bool = false;
const INFO: bool = true;

// this may speed up more complex grammar but slows down simple ones (by 10%)
const PREDICTED_SYM_FILTER: bool = false;

macro_rules! debug {
    ($($arg:tt)*) => {
        if DEBUG {
            println!($($arg)*);
        }
    }
}

macro_rules! info {
    ($($arg:tt)*) => {
        if INFO {
            println!($($arg)*);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Item {
    data: u64,
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
    const NULL: Self = Item { data: 0 };

    fn new(rule: RuleIdx, start: usize) -> Self {
        Item {
            data: rule.as_index() as u64 | ((start as u64) << 32),
        }
    }

    fn rule_idx(&self) -> RuleIdx {
        RuleIdx::from_index(self.data as u32)
    }

    fn start_pos(&self) -> usize {
        (self.data >> 32) as usize
    }

    fn advance_dot(&self) -> Self {
        Item {
            data: self.data + 1,
        }
    }
}

impl SimpleHash for Item {
    fn simple_hash(&self) -> u32 {
        (self.rule_idx().as_index() as u32)
            .wrapping_mul(16315967)
            .wrapping_add((self.start_pos() as u32).wrapping_mul(33398653))
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

    #[inline(always)]
    fn insert(&mut self, item: T) {
        let mask = item.mask64();
        if (self.hash & mask) != 0 && self.items.contains(&item) {
            return;
        }
        self.hash |= mask;
        self.items.push(item);
    }

    #[inline(always)]
    fn contains(&self, item: T) -> bool {
        if (item.mask64() & self.hash) == 0 {
            false
        } else {
            self.items.contains(&item)
        }
    }

    #[inline(always)]
    fn should_insert(&mut self, item: T) -> bool {
        if !PREDICTED_SYM_FILTER {
            true
        } else {
            if self.contains(item) {
                false
            } else {
                self.insert(item);
                true
            }
        }
    }
}

#[derive(Default)]
struct Scratch {
    row_start: usize,
    row_end: usize,
    items: Vec<Item>,
    predicated_syms: SimpleSet<CSymIdx>,
    speculative: bool,
}

struct RowInfo {
    byte: u8,
    token_idx: usize,
    #[allow(dead_code)]
    commit_item: Item,
}

pub struct Parser {
    grammar: CGrammar,
    scratch: Scratch,
    captures: Vec<(String, Vec<u8>)>,
    rows: Vec<Row>,
    row_infos: Vec<RowInfo>,
    stats: Stats,
    is_accepting: bool,
    last_collapse: usize,
    speculative: bool,
    token_idx: usize,
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
        // SAFETY: we just ensured that there is enough space
        unsafe {
            self.items.as_mut_ptr().add(self.row_end).write(item);
        }
        // self.items[self.row_end] = item;
        self.row_end += 1;
    }

    #[inline(always)]
    fn add_unique(&mut self, item: Item, grm: &CGrammar, info: &str) {
        if !self.items[self.row_start..self.row_end].contains(&item) {
            if !self.speculative {
                debug!("      addu: {} ({})", item_to_string(grm, &item), info);
            }
            self.just_add(item);
        }
    }
}

impl Parser {
    pub fn new(grammar: CGrammar) -> Self {
        let start = grammar.start();
        let mut r = Parser {
            grammar,
            rows: vec![],
            row_infos: vec![],
            captures: vec![],
            scratch: Scratch::default(),
            stats: Stats::default(),
            is_accepting: false,
            last_collapse: 0,
            speculative: false,
            token_idx: 0,
        };
        for rule in r.grammar.rules_of(start).to_vec() {
            r.scratch.add_unique(Item::new(rule, 0), &r.grammar, "init");
        }
        debug!("initial push");
        let _ = r.push_row(r.scratch.row_start, 0);
        r
    }

    pub fn is_accepting(&self) -> bool {
        self.is_accepting
    }

    fn item_to_string(&self, item: &Item) -> String {
        item_to_string(&self.grammar, item)
    }

    pub fn print_row(&self, row_idx: usize) {
        let row = &self.rows[row_idx];
        println!("row {}", row_idx);
        for i in row.item_indices() {
            println!("{}", self.item_to_string(&self.scratch.items[i]));
        }
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    fn pop_row_infos(&mut self, n: usize) {
        self.assert_non_trie();
        unsafe { self.row_infos.set_len(self.row_infos.len() - n) }
        self.pop_rows(n);
    }

    fn pop_rows(&mut self, n: usize) {
        unsafe { self.rows.set_len(self.rows.len() - n) }
        // self.rows.drain(self.rows.len() - n..);
    }

    #[allow(dead_code)]
    pub fn print_stats(&mut self) {
        println!("stats: {:?}", self.stats);
        self.stats = Stats::default();
    }

    fn assert_non_trie(&self) {
        assert!(!self.speculative);
        assert!(self.num_rows() == self.row_infos.len());
    }

    pub fn get_bytes(&self) -> Vec<u8> {
        self.assert_non_trie();
        self.row_infos.iter().skip(1).map(|ri| ri.byte).collect()
    }

    fn item_lhs(&self, item: &Item) -> CSymIdx {
        self.grammar.sym_idx_of(item.rule_idx())
    }

    fn item_sym_data(&self, item: &Item) -> &CSymbol {
        self.grammar.sym_data(self.item_lhs(item))
    }

    pub fn apply_tokens(
        &mut self,
        trie: &TokTrie,
        tokens: &[TokenId],
        mut num_skip: usize,
    ) -> &'static str {
        // this is unused!
        self.assert_non_trie();
        let mut byte_idx = 1; // row_infos[0] has just the 0 byte
        let mut tok_idx = 0;
        for t in tokens {
            for b in trie.token(*t).iter() {
                if num_skip > 0 {
                    num_skip -= 1;
                    continue;
                }

                if byte_idx >= self.row_infos.len() {
                    if self.scan(*b) == ParseResult::Reject {
                        return "parse reject";
                    }
                    if byte_idx >= self.row_infos.len() {
                        return "hidden item";
                    }
                }
                let info = &mut self.row_infos[byte_idx];
                if info.byte != *b {
                    println!("byte mismatch: {} != {} at {}", info.byte, b, byte_idx);
                    return "static reject";
                }
                info.token_idx = tok_idx;
                byte_idx += 1;
            }
            tok_idx += 1;
        }
        while byte_idx < self.row_infos.len() {
            self.row_infos[byte_idx].token_idx = tok_idx;
            byte_idx += 1;
        }
        self.token_idx = tok_idx;
        return "";
    }

    pub fn filter_max_tokens(&mut self) {
        let mut dst = 0;

        self.row_infos.push(RowInfo {
            byte: 0,
            commit_item: Item::NULL,
            token_idx: self.token_idx,
        });

        for idx in 0..self.rows.len() {
            let range = self.rows[idx].item_indices();
            self.rows[idx].first_item = dst;
            for i in range {
                let item = self.scratch.items[i];
                let sym_data = self.item_sym_data(&item);
                let max_tokens = sym_data.props.max_tokens;
                if max_tokens != usize::MAX {
                    let start_token_idx = self.row_infos[item.start_pos() + 1].token_idx;
                    if self.token_idx - start_token_idx >= max_tokens {
                        debug!(
                            "  remove: {}-{} {}",
                            self.token_idx,
                            start_token_idx,
                            self.item_to_string(&item)
                        );
                        continue;
                    }
                }
                self.scratch.items[dst] = item;
                dst += 1;
            }
            self.rows[idx].last_item = dst;
        }

        self.row_infos.pop();
    }

    pub fn force_bytes(&mut self) -> Vec<u8> {
        self.assert_non_trie();
        let mut bytes = vec![];
        while let Some(b) = self.forced_byte() {
            let res = self.scan(b);
            if res == ParseResult::Reject {
                // shouldn't happen?
                break;
            }
            bytes.push(b);
        }
        bytes
    }

    fn curr_row(&self) -> &Row {
        &self.rows[self.rows.len() - 1]
    }

    pub fn model_variables(&self) -> Vec<ModelVariable> {
        let mut vars = vec![];
        for i in self.curr_row().item_indices() {
            let item = self.scratch.items[i];
            let sym = self.grammar.sym_idx_at(item.rule_idx());
            let sym_data = self.grammar.sym_data(sym);
            if let Some(ref mv) = sym_data.props.model_variable {
                if !vars.contains(mv) {
                    vars.push(mv.clone());
                }
            }
        }
        vars
    }

    fn forced_byte(&self) -> Option<u8> {
        if self.is_accepting {
            // we're not forced when in accepting state
            return None;
        }

        let mut byte_sym = None;
        for i in self.curr_row().item_indices() {
            let item = self.scratch.items[i];
            let sym = self.grammar.sym_idx_at(item.rule_idx());
            if self.grammar.is_terminal(sym) {
                if self.grammar.is_single_byte_terminal(sym) {
                    if byte_sym == None || byte_sym == Some(sym) {
                        byte_sym = Some(sym);
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
        }

        if let Some(s) = byte_sym {
            let r = self.grammar.terminal_byteset(s).single_byte();
            assert!(r.is_some());
            r
        } else {
            None
        }
    }

    pub fn hide_item(&mut self, sym: CSymIdx, row_idx: usize) -> ParseResult {
        info!("hide_item: {} {}", self.grammar.sym_data(sym).name, row_idx);

        let row_range = self.rows[row_idx].item_indices();
        let last_byte = self.row_infos[row_idx].byte;
        let agenda_ptr = row_range.start;
        let to_pop = self.num_rows() - row_idx;
        assert!(to_pop > 0);
        self.pop_row_infos(to_pop);
        assert!(self.num_rows() == row_idx);

        let mut items_to_add = vec![];
        for idx in row_range {
            let item = self.scratch.items[idx];
            //info!("  => now: {}", item_to_string(&self.grammar, &item));
            if self.grammar.sym_idx_at(item.rule_idx()) == sym {
                info!(
                    "  => add: {}",
                    item_to_string(&self.grammar, &item.advance_dot())
                );
                items_to_add.push(item.advance_dot());
            }
        }

        // we remove everything from the current row before adding the entries
        self.scratch.new_row(agenda_ptr);
        for item in items_to_add {
            self.scratch.add_unique(item, &self.grammar, "hide");
        }

        self.push_row(agenda_ptr, last_byte)
    }

    #[inline(always)]
    pub fn scan(&mut self, b: u8) -> ParseResult {
        let row_idx = self.rows.len() - 1;
        let last = self.rows[row_idx].last_item;
        let mut i = self.rows[row_idx].first_item;
        let n = last - i;
        self.scratch.ensure_items(last + n + 100);

        let allowed = self.grammar.terminals_by_byte(b);

        self.scratch.new_row(last);

        if !self.speculative {
            debug!("  scan: {:?}", b as char);
        }

        while i < last {
            let item = self.scratch.items[i];
            let idx = self.grammar.sym_idx_at(item.rule_idx()).as_index();
            // idx == 0 => completed
            if idx < allowed.len() && allowed[idx] {
                self.scratch.just_add(item.advance_dot());
            }
            i += 1;
        }
        self.push_row(self.scratch.row_start, b)
    }

    pub fn captures(&self) -> &[(String, Vec<u8>)] {
        &self.captures
    }

    #[inline(always)]
    fn push_row(&mut self, mut agenda_ptr: usize, byte: u8) -> ParseResult {
        let curr_idx = self.rows.len();
        let mut commit_item = Item::NULL;

        self.scratch.predicated_syms.clear();

        self.stats.rows += 1;
        self.is_accepting = false;

        while agenda_ptr < self.scratch.row_end {
            let mut item = self.scratch.items[agenda_ptr];
            agenda_ptr += 1;
            if !self.speculative {
                debug!("    agenda: {}", self.item_to_string(&item));
            }

            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_at(rule);

            if after_dot == CSymIdx::NULL {
                let flags = self.grammar.sym_flags_of(rule);
                let lhs = self.grammar.sym_idx_of(item.rule_idx());
                // complete
                self.is_accepting = self.is_accepting || lhs == self.grammar.start();

                if !self.speculative && flags.capture() {
                    let var_name = self
                        .grammar
                        .sym_data(lhs)
                        .props
                        .capture_name
                        .as_ref()
                        .unwrap();
                    let mut bytes = Vec::new();
                    if item.start_pos() + 1 < curr_idx {
                        bytes = self.row_infos[item.start_pos() + 1..curr_idx]
                            .iter()
                            .map(|ri| ri.byte)
                            .collect::<Vec<_>>();
                    }
                    bytes.push(byte);
                    debug!(
                        "      capture: {} {:?}",
                        var_name,
                        String::from_utf8_lossy(&bytes)
                    );
                    self.captures.push((var_name.clone(), bytes));
                }

                if flags.commit_point() {
                    // TODO do we need to remove possible scans?
                    for ptr in agenda_ptr..self.scratch.row_end {
                        let next_item = self.scratch.items[ptr];
                        let next_rule = next_item.rule_idx();
                        // is it earlier, complete, and commit point?
                        if next_item.start_pos() < item.start_pos()
                            && self.grammar.sym_idx_at(next_rule) == CSymIdx::NULL
                            && self.grammar.sym_flags_of(next_rule).commit_point()
                        {
                            // if so, use it
                            item = next_item;
                        }
                    }
                    self.scratch.row_end = agenda_ptr;
                    self.scratch.items[agenda_ptr - 1] = item;
                    commit_item = item;
                    if !self.speculative {
                        debug!("  commit point: {}", self.item_to_string(&item));
                    }
                    if !self.speculative && flags.hidden() {
                        return self.hide_item(lhs, item.start_pos());
                    }
                }

                if item.start_pos() < curr_idx {
                    // if item.start_pos() == curr_idx, then we handled it below in the nullable check
                    for i in self.rows[item.start_pos()].item_indices() {
                        let item = self.scratch.items[i];
                        if self.grammar.sym_idx_at(item.rule_idx()) == lhs {
                            self.scratch
                                .add_unique(item.advance_dot(), &self.grammar, "complete");
                        }
                    }
                }
            } else {
                let sym_data = self.grammar.sym_data(after_dot);
                // if no max_tokens, no point looking in row_infos
                // if start_pos() is recent enough, not to make it into row_infos, also no point checking
                // otherwise, if we exceeded max_tokens, don't add anything
                // if sym_data.props.max_tokens != usize::MAX
                //     && item.start_pos() < self.row_infos.len()
                //     && self.token_idx - self.row_infos[item.start_pos()].token_idx
                //         >= sym_data.props.max_tokens
                // {
                //     continue;
                // }
                if sym_data.is_nullable {
                    self.scratch
                        .add_unique(item.advance_dot(), &self.grammar, "null");
                }
                if self.scratch.predicated_syms.should_insert(after_dot) {
                    for rule in &sym_data.rules {
                        let new_item = Item::new(*rule, curr_idx);
                        self.scratch.add_unique(new_item, &self.grammar, "predict");
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

        if !self.speculative {
            self.row_infos.drain((self.rows.len() - 1)..);
            self.row_infos.push(RowInfo {
                byte,
                commit_item,
                token_idx: self.token_idx,
            });
        }

        if self.is_accepting {
            ParseResult::Accept
        } else {
            ParseResult::Continue
        }
    }
}

impl Recognizer for Parser {
    fn pop_bytes(&mut self, num: usize) {
        self.pop_rows(num);
    }

    fn collapse(&mut self) {
        // this actually means "commit" - can no longer backtrack past this point

        if false {
            for idx in self.last_collapse..self.num_rows() {
                self.print_row(idx);
            }
        }
        self.last_collapse = self.num_rows();
    }

    fn special_allowed(&mut self, tok: SpecialToken) -> bool {
        if false {
            self.print_row(self.num_rows() - 1);
            println!(
                "model vars: accpt={} {:?}",
                self.is_accepting(),
                self.model_variables()
            );
        }

        if self
            .model_variables()
            .contains(&ModelVariable::SpecialToken(tok))
        {
            true
        } else if tok == SpecialToken::EndOfSentence {
            self.is_accepting()
        } else {
            false
        }
    }

    fn trie_started(&mut self) {
        // println!("trie_started: rows={} infos={}", self.num_rows(), self.row_infos.len());
        assert!(self.speculative == false);
        assert!(self.row_infos.len() == self.num_rows());
        self.speculative = true;
        self.scratch.speculative = true;
    }

    fn trie_finished(&mut self) {
        // println!("trie_finished: rows={} infos={}", self.num_rows(), self.row_infos.len());
        assert!(self.speculative == true);
        assert!(self.row_infos.len() <= self.num_rows());
        // clean up stack
        self.pop_rows(self.num_rows() - self.row_infos.len());
        self.speculative = false;
        self.scratch.speculative = false;
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        let res = self.scan(byte);
        if res == ParseResult::Reject {
            false
        } else {
            true
        }
    }
}

fn item_to_string(g: &CGrammar, item: &Item) -> String {
    format!(
        "{} @{}",
        g.rule_to_string(item.rule_idx()),
        item.start_pos(),
    )
}
