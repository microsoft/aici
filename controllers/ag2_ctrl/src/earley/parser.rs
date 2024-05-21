use std::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::Range,
    rc::Rc,
    vec,
};

use aici_abi::{
    toktree::{Recognizer, SpecialToken, TokTrie},
    TokenId,
};
use anyhow::{bail, Result};

use super::{
    grammar::{CGrammar, CSymIdx, CSymbol, ModelVariable, RuleIdx},
    lex::Lexeme,
};

const DEBUG: bool = false;
const INFO: bool = true;
const MAX_ROW: usize = 100;

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

// These are only tracked in definitive mode
#[derive(Debug, Clone)]
struct ItemProps {
    hidden_start: usize,
}

impl Display for ItemProps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.hidden_start == usize::MAX {
            write!(f, "")
        } else {
            write!(f, "(hidden_start {})", self.hidden_start)
        }
    }
}

impl Default for ItemProps {
    fn default() -> Self {
        ItemProps {
            hidden_start: usize::MAX,
        }
    }
}

impl ItemProps {
    fn merge(&mut self, other: ItemProps) {
        self.hidden_start = self.hidden_start.min(other.hidden_start);
    }
}

#[derive(Debug, Default)]
pub struct Stats {
    pub rows: usize,
    pub empty_rows: usize,
    pub nontrivial_scans: usize,
    pub scan_items: usize,
    pub all_items: usize,
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

struct Scratch {
    grammar: Rc<CGrammar>,
    row_start: usize,
    row_end: usize,
    items: Vec<Item>,
    item_props: Vec<ItemProps>,
    definitive: bool,
}

struct RowInfo {
    lexeme: Lexeme,
    token_idx: usize,
    #[allow(dead_code)]
    commit_item: Item,
}

pub struct Parser {
    grammar: Rc<CGrammar>,
    scratch: Scratch,
    captures: Vec<(String, Vec<u8>)>,
    rows: Vec<Row>,
    row_infos: Vec<RowInfo>,
    stats: Stats,
    last_collapse: usize,
    token_idx: usize,
}

impl Scratch {
    fn new(grammar: Rc<CGrammar>) -> Self {
        Scratch {
            grammar,
            row_start: 0,
            row_end: 0,
            items: vec![],
            item_props: vec![],
            definitive: true,
        }
    }

    fn new_row(&mut self, pos: usize) {
        self.row_start = pos;
        self.row_end = pos;
    }

    fn row_len(&self) -> usize {
        self.row_end - self.row_start
    }

    fn work_row(&self) -> Row {
        Row {
            first_item: self.row_start,
            last_item: self.row_end,
        }
    }

    fn hidden_start(&self, r: &Row) -> usize {
        r.item_indices()
            .map(|i| self.item_props[i].hidden_start)
            .min()
            .unwrap_or(usize::MAX)
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
    fn merge_item_origin(&mut self, target_item_idx: usize, origin_item_idx: usize) {
        let origin = self.item_props[origin_item_idx].clone();
        self.item_props[target_item_idx].merge(origin);
    }

    #[inline(always)]
    fn just_add(&mut self, item: Item, origin_item_idx: usize, info: &str) {
        self.ensure_items(self.row_end + 1);
        // SAFETY: we just ensured that there is enough space
        unsafe {
            self.items.as_mut_ptr().add(self.row_end).write(item);
        }
        // self.items[self.row_end] = item;
        if self.definitive {
            if self.item_props.len() <= self.row_end {
                self.item_props.push(ItemProps::default());
            } else {
                self.item_props[self.row_end] = ItemProps::default();
            }
            self.merge_item_origin(self.row_end, origin_item_idx);

            debug!(
                "      addu: {} ({})",
                self.item_to_string(self.row_end),
                info
            );
        }
        self.row_end += 1;
    }

    #[inline(always)]
    fn find_item(&self, item: Item) -> Option<usize> {
        self.items[self.row_start..self.row_end]
            .iter()
            .position(|&x| x == item)
            .map(|x| x + self.row_start)
    }

    fn set_hidden_start(&mut self, item: Item, hidden_start: usize) {
        let idx = self.find_item(item).unwrap();
        self.item_props[idx].hidden_start =
            std::cmp::min(self.item_props[idx].hidden_start, hidden_start);
        debug!(
            "      hidden: {} {}",
            hidden_start,
            self.item_to_string(idx),
        );
    }

    #[inline(always)]
    fn add_unique(&mut self, item: Item, origin_item_idx: usize, info: &str) {
        if let Some(idx) = self.find_item(item) {
            if self.definitive {
                self.merge_item_origin(idx, origin_item_idx);
            }
        } else {
            self.just_add(item, origin_item_idx, info);
        }
    }

    fn item_to_string(&self, idx: usize) -> String {
        let r = item_to_string(&self.grammar, &self.items[idx]);
        if self.definitive {
            let props = &self.item_props[idx];
            format!("{} {}", r, props)
        } else {
            r
        }
    }
}

impl Parser {
    pub fn new(grammar: CGrammar) -> Self {
        let start = grammar.start();
        let grammar = Rc::new(grammar);
        let scratch = Scratch::new(Rc::clone(&grammar));
        let mut r = Parser {
            grammar,
            rows: vec![],
            row_infos: vec![],
            captures: vec![],
            scratch,
            stats: Stats::default(),
            last_collapse: 0,
            token_idx: 0,
        };
        info!("new parser");
        for rule in r.grammar.rules_of(start).to_vec() {
            r.scratch.add_unique(Item::new(rule, 0), 0, "init");
        }
        debug!("initial push");
        let _ = r.push_row(r.scratch.row_start, Lexeme::bogus());
        r
    }

    pub fn is_accepting(&self) -> bool {
        for idx in self.curr_row().item_indices() {
            let item = self.scratch.items[idx];
            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_at(rule);
            if after_dot == CSymIdx::NULL {
                let lhs = self.grammar.sym_idx_of(item.rule_idx());
                if lhs == self.grammar.start() {
                    return true;
                }
            }
        }
        false
    }

    fn item_to_string(&self, idx: usize) -> String {
        self.scratch.item_to_string(idx)
    }

    pub fn print_row(&self, row_idx: usize) {
        let row = &self.rows[row_idx];
        println!("row {}", row_idx);
        for i in row.item_indices() {
            println!("{}", self.item_to_string(i));
        }
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
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

    fn assert_definitive(&self) {
        assert!(self.scratch.definitive);
        assert!(self.num_rows() == self.row_infos.len());
    }

    pub fn get_bytes(&self) -> Vec<u8> {
        self.assert_definitive();
        self.row_infos
            .iter()
            .map(|ri| ri.lexeme.visible_bytes())
            .collect::<Vec<_>>()
            .concat()
    }

    fn item_lhs(&self, item: &Item) -> CSymIdx {
        self.grammar.sym_idx_of(item.rule_idx())
    }

    fn item_sym_data(&self, item: &Item) -> &CSymbol {
        self.grammar.sym_data(self.item_lhs(item))
    }

    pub fn hidden_start(&self) -> usize {
        self.scratch.hidden_start(&self.curr_row())
    }

    pub fn temperature(&self) -> f32 {
        let mut temp = 0.0f32;
        for i in self.curr_row().item_indices() {
            let item = self.scratch.items[i];
            let data = self.grammar.sym_data_at(item.rule_idx());
            if data.is_terminal {
                temp = temp.max(data.props.temperature);
            }
        }
        temp
    }

    pub fn apply_tokens(
        &mut self,
        trie: &TokTrie,
        tokens: &[TokenId],
        mut num_skip: usize,
    ) -> Result<&'static str> {
        // this is unused!
        self.assert_definitive();
        let mut lexeme_idx = 0;
        let mut lexeme_byte_idx = 0;
        let mut tok_idx = 0;
        debug!("apply_tokens: {:?}", tokens);
        for t in tokens {
            for b in trie.token(*t).iter() {
                if num_skip > 0 {
                    num_skip -= 1;
                    continue;
                }

                // scan to the next byte
                while lexeme_idx < self.row_infos.len()
                    && lexeme_byte_idx >= self.row_infos[lexeme_idx].lexeme.num_visible_bytes()
                {
                    self.row_infos[lexeme_idx].token_idx = tok_idx;
                    lexeme_byte_idx = 0;
                    lexeme_idx += 1;
                }

                if lexeme_idx >= self.row_infos.len() {
                    if !self.try_push_byte(*b) {
                        return Ok("parse reject");
                    }
                    if lexeme_idx >= self.row_infos.len() {
                        return Ok("hidden item");
                    }
                    lexeme_byte_idx = 0;
                    let item_count = self.curr_row().item_indices().count();
                    if item_count > MAX_ROW {
                        bail!(
                            "Current row has {} items; max is {}; consider making your grammar left-recursive if it's right-recursive",
                            item_count,
                            MAX_ROW,
                        );
                    }
                }

                let info = &mut self.row_infos[lexeme_idx];
                let lb = info.lexeme.visible_bytes()[lexeme_byte_idx];
                if lb != *b {
                    println!("byte mismatch: {} != {} at {}", lb, b, lexeme_idx);
                    return Ok("static reject");
                }
                lexeme_byte_idx += 1;
            }
            tok_idx += 1;
        }
        while lexeme_idx < self.row_infos.len() {
            self.row_infos[lexeme_idx].token_idx = tok_idx;
            lexeme_idx += 1;
        }
        self.token_idx = tok_idx;
        return Ok("");
    }

    pub fn filter_max_tokens(&mut self) {
        self.assert_definitive();

        let mut dst = 0;

        self.row_infos.push(RowInfo {
            lexeme: Lexeme::bogus(),
            commit_item: Item::NULL,
            token_idx: self.token_idx,
        });

        for idx in 0..self.rows.len() {
            let range = self.rows[idx].item_indices();
            self.rows[idx].first_item = dst;
            for i in range {
                let item = self.scratch.items[i];
                let item_props = &self.scratch.item_props[i];
                let sym_data = self.item_sym_data(&item);
                let max_tokens = sym_data.props.max_tokens;
                if max_tokens != usize::MAX {
                    let start_token_idx = self.row_infos[item.start_pos() + 1].token_idx;
                    if self.token_idx - start_token_idx >= max_tokens {
                        debug!(
                            "  remove: {}-{} {}",
                            self.token_idx,
                            start_token_idx,
                            self.item_to_string(i)
                        );
                        continue;
                    }
                }
                self.scratch.items[dst] = item;
                self.scratch.item_props[dst] = item_props.clone();
                dst += 1;
            }
            self.rows[idx].last_item = dst;
        }

        self.row_infos.pop();
    }

    pub fn force_bytes(&mut self) -> Vec<u8> {
        self.assert_definitive();
        debug!("force_bytes");
        let mut bytes = vec![];
        while let Some(b) = self.forced_byte() {
            if !self.try_push_byte(b) {
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
            let sym_data = self.grammar.sym_data_at(item.rule_idx());
            if let Some(ref mv) = sym_data.props.model_variable {
                if !vars.contains(mv) {
                    vars.push(mv.clone());
                }
            }
        }
        vars
    }

    fn forced_byte(&mut self) -> Option<u8> {
        if self.is_accepting() {
            // we're not forced when in accepting state
            return None;
        }

        let mut byte_sym = None;
        for b in 0..=255 {
            if self.try_push_byte(b) {
                self.pop_bytes(1);
                if byte_sym.is_some() {
                    return None; // more than one option
                } else {
                    byte_sym = Some(b);
                }
            }
        }
        byte_sym
    }

    pub fn scan_model_variable(&mut self, mv: ModelVariable) -> bool {
        if self.scratch.definitive {
            debug!("  scan mv: {:?}", mv);
        }

        self.scratch.new_row(self.curr_row().last_item);

        for idx in self.curr_row().item_indices() {
            let item = self.scratch.items[idx];
            let sym_data = self.grammar.sym_data_at(item.rule_idx());
            if let Some(ref mv2) = sym_data.props.model_variable {
                if mv == *mv2 {
                    self.scratch
                        .add_unique(item.advance_dot(), idx, "scan_model_variable");
                }
            }
        }

        if self.scratch.row_len() == 0 {
            false
        } else {
            self.push_row(self.scratch.row_start, Lexeme::bogus())
        }
    }

    pub fn scan(&mut self, lexeme: Lexeme) -> bool {
        let row_idx = self.rows.len() - 1;
        let last = self.rows[row_idx].last_item;
        let mut i = self.rows[row_idx].first_item;
        let n = last - i;
        self.scratch.ensure_items(last + n + 100);
        self.scratch.new_row(last);

        if self.scratch.definitive {
            debug!("  scan: {:?}", self.grammar.dbg_lexeme(&lexeme));
        }

        let trg = self.grammar.lexeme_to_sym_idx(lexeme.idx);

        while i < last {
            let item = self.scratch.items[i];
            let idx = self.grammar.sym_idx_at(item.rule_idx());
            if idx == trg {
                self.scratch.just_add(item.advance_dot(), i, "scan");
            }
            i += 1;
        }
        self.push_row(self.scratch.row_start, lexeme)
    }

    pub fn captures(&self) -> &[(String, Vec<u8>)] {
        &self.captures
    }

    #[inline(always)]
    fn push_row(&mut self, mut agenda_ptr: usize, lexeme: Lexeme) -> bool {
        let curr_idx = self.rows.len();
        let mut commit_item = Item::NULL;

        self.stats.rows += 1;

        while agenda_ptr < self.scratch.row_end {
            let mut item_idx = agenda_ptr;
            let mut item = self.scratch.items[agenda_ptr];
            agenda_ptr += 1;
            if self.scratch.definitive {
                debug!("    agenda: {}", self.item_to_string(item_idx));
            }

            let rule = item.rule_idx();
            let after_dot = self.grammar.sym_idx_at(rule);

            if after_dot == CSymIdx::NULL {
                let flags = self.grammar.sym_flags_of(rule);
                let lhs = self.grammar.sym_idx_of(rule);

                if self.scratch.definitive && flags.capture() {
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
                            .map(|ri| ri.lexeme.visible_bytes())
                            .collect::<Vec<_>>()
                            .concat();
                    }
                    bytes.extend_from_slice(lexeme.visible_bytes());
                    debug!(
                        "      capture: {} {:?}",
                        var_name,
                        String::from_utf8_lossy(&bytes)
                    );
                    self.captures.push((var_name.clone(), bytes));
                }

                if item.start_pos() < curr_idx {
                    // if item.start_pos() == curr_idx, then we handled it below in the nullable check
                    for i in self.rows[item.start_pos()].item_indices() {
                        let item = self.scratch.items[i];
                        if self.grammar.sym_idx_at(item.rule_idx()) == lhs {
                            self.scratch.add_unique(item.advance_dot(), i, "complete");
                        }
                    }
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
                            item_idx = ptr;
                        }
                    }
                    self.scratch.row_end = agenda_ptr;
                    self.scratch.items[agenda_ptr - 1] = item;
                    if self.scratch.definitive {
                        self.scratch.item_props[agenda_ptr - 1] =
                            self.scratch.item_props[item_idx].clone();
                    }
                    item_idx = agenda_ptr - 1;
                    commit_item = item;
                    if self.scratch.definitive {
                        debug!("  commit point: {}", self.item_to_string(item_idx));
                        if flags.hidden() {
                            panic!("hidden item");
                            // return self.hide_item(lhs, item.start_pos());
                        }
                    }
                }
            } else {
                let sym_data = self.grammar.sym_data(after_dot);
                if sym_data.is_nullable {
                    self.scratch
                        .add_unique(item.advance_dot(), item_idx, "null");
                }
                for rule in &sym_data.rules {
                    let new_item = Item::new(*rule, curr_idx);
                    self.scratch.add_unique(new_item, item_idx, "predict");
                }
                if self.scratch.definitive && sym_data.props.hidden {
                    for rule in &sym_data.rules {
                        let new_item = Item::new(*rule, curr_idx);
                        self.scratch.set_hidden_start(new_item, curr_idx);
                    }
                }
            }
        }

        let row_len = self.scratch.row_len();

        if row_len == 0 {
            false
        } else {
            self.stats.all_items += row_len;

            self.rows.push(self.scratch.work_row());

            if self.scratch.definitive {
                self.row_infos.drain((self.rows.len() - 1)..);
                self.row_infos.push(RowInfo {
                    lexeme,
                    commit_item,
                    token_idx: self.token_idx,
                });
            }

            true
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
        assert!(self.scratch.definitive);
        assert!(self.row_infos.len() == self.num_rows());
        self.scratch.definitive = false;
    }

    fn trie_finished(&mut self) {
        // println!("trie_finished: rows={} infos={}", self.num_rows(), self.row_infos.len());
        assert!(self.scratch.definitive == false);
        assert!(self.row_infos.len() <= self.num_rows());
        // clean up stack
        self.pop_rows(self.num_rows() - self.row_infos.len());
        self.scratch.definitive = true;
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        todo!()
        // self.scan(byte)
    }
}

fn item_to_string(g: &CGrammar, item: &Item) -> String {
    format!(
        "{} @{}",
        g.rule_to_string(item.rule_idx()),
        item.start_pos(),
    )
}
