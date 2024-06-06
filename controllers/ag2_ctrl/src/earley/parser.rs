use std::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::Range,
    rc::Rc,
    vec,
};

use aici_abi::{
    svob::SimpleVob,
    toktree::{Recognizer, SpecialToken, TokTrie},
    TokenId,
};
use anyhow::{bail, ensure, Result};

use crate::earley::lexer::Lexer;

use super::{
    grammar::{CGrammar, CSymIdx, CSymbol, ModelVariable, RuleIdx},
    lexer::{Lexeme, LexemeIdx, LexerSpec, StateID},
};

const TRACE: bool = false;
const DEBUG: bool = true;
const INFO: bool = true;
const MAX_ROW: usize = 100;

macro_rules! trace {
    ($($arg:tt)*) => {
        if TRACE {
            println!($($arg)*);
        }
    }
}

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
    allowed_lexemes: SimpleVob,
}

impl Row {
    fn item_indices(&self) -> Range<usize> {
        self.first_item..self.last_item
    }
}

impl Item {
    #[allow(dead_code)]
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
    token_idx_start: usize,
    token_idx_stop: usize,
    max_tokens: usize,
}

impl RowInfo {
    fn apply_token_idx(&mut self, idx: usize) {
        self.token_idx_start = self.token_idx_start.min(idx);
        self.token_idx_stop = self.token_idx_stop.max(idx);
    }
}

// State transition is:
// if (next_lexeme, next_lexer_state) := lexer(top.lexer_state, next_byte) {
//     row_idx = scan(top.row_idx, next_lexeme)
//     push(LexerState { row_idx, next_byte, next_lexer_state })
// }
#[derive(Clone, Copy)]
struct LexerState {
    row_idx: u32,
    lexer_state: StateID, // state after consuming byte
    byte: u8,
    use_byte: bool,
}

pub struct Parser {
    lexer: Lexer,
    grammar: Rc<CGrammar>,
    scratch: Scratch,
    trie_lexer_stack: usize,
    captures: Vec<(String, Vec<u8>)>,
    lexer_stack: Vec<LexerState>,
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

    fn work_row(&self, allowed_lexemes: SimpleVob) -> Row {
        Row {
            first_item: self.row_start,
            last_item: self.row_end,
            allowed_lexemes,
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

macro_rules! ensure_internal {
    ($cond:expr, $msg:expr) => {
        ensure!($cond, "Internal error: {}", $msg)
    };
}

impl Parser {
    pub fn new(grammar: CGrammar) -> Result<Self> {
        let start = grammar.start();
        let lexer = Lexer::from(grammar.lexer_spec().clone());
        let grammar = Rc::new(grammar);
        let scratch = Scratch::new(Rc::clone(&grammar));
        let lexer_state = lexer.file_start_state();
        let mut r = Parser {
            grammar,
            lexer,
            trie_lexer_stack: usize::MAX,
            rows: vec![],
            row_infos: vec![],
            captures: vec![],
            scratch,
            stats: Stats::default(),
            last_collapse: 0,
            token_idx: 0,
            lexer_stack: vec![LexerState {
                row_idx: 0,
                lexer_state,
                byte: 0,
                use_byte: false,
            }],
        };
        info!("new parser");
        for rule in r.grammar.rules_of(start).to_vec() {
            r.scratch.add_unique(Item::new(rule, 0), 0, "init");
        }
        debug!("initial push");
        let _ = r.push_row(r.scratch.row_start, &Lexeme::bogus());
        ensure_internal!(
            r.num_rows() == 1 && r.rows.len() == 1,
            "initial push failed"
        );
        r.assert_definitive();
        Ok(r)
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
        println!("row {}; lexer_stack={}", row_idx, self.lexer_stack.len());

        println!(
            "  allowed: {}",
            self.lexer_spec().dbg_lexeme_set(&row.allowed_lexemes)
        );

        if row_idx == 0 {
            println!("  no lexeme on first row")
        } else if row_idx < self.row_infos.len() {
            let info = &self.row_infos[row_idx];
            println!("  lexeme: {}", self.lexer_spec().dbg_lexeme(&info.lexeme));
        } else {
            println!("  speculative");
        }
        for i in row.item_indices() {
            println!("  {}", self.item_to_string(i));
        }
    }

    #[inline(always)]
    fn lexer_state(&self) -> LexerState {
        self.lexer_stack[self.lexer_stack.len() - 1]
    }

    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        self.lexer_state().row_idx as usize + 1
    }

    fn pop_lexer_states(&mut self, n: usize) {
        assert!(self.lexer_stack.len() > n);
        unsafe { self.lexer_stack.set_len(self.lexer_stack.len() - n) }
    }

    #[allow(dead_code)]
    pub fn print_stats(&mut self) {
        println!("stats: {:?}", self.stats);
        self.stats = Stats::default();
    }

    fn assert_definitive(&self) {
        assert!(self.scratch.definitive);
        if self.num_rows() != self.row_infos.len() {
            panic!(
                "num_rows={} row_infos={}",
                self.num_rows(),
                self.row_infos.len()
            );
        }
    }

    fn get_bytes_and_lexeme_indices(&self) -> (Vec<usize>, Vec<u8>) {
        self.assert_definitive();
        let mut indices = vec![];
        let mut allbytes = vec![];
        trace!("get_bytes:");
        for (idx, ri) in self.row_infos.iter().enumerate() {
            trace!("  lexeme: {}", self.lexer_spec().dbg_lexeme(&ri.lexeme));
            let mut bytes = ri.lexeme.visible_bytes().to_vec();
            if bytes.is_empty() && idx == self.num_rows() - 1 {
                bytes = self.curr_row_bytes();
                trace!("    bytes: {:?}", String::from_utf8_lossy(&bytes));
            };
            indices.extend((0..bytes.len()).map(|_| idx));
            allbytes.extend_from_slice(&bytes);
        }
        (indices, allbytes)
    }

    pub fn get_bytes(&self) -> Vec<u8> {
        self.get_bytes_and_lexeme_indices().1
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
        debug!("apply_tokens: {:?} {}", tokens, self.lexer_stack.len());
        self.assert_definitive();
        let mut tok_idx = 0;
        // reset token_idx
        for ri in self.row_infos.iter_mut() {
            ri.token_idx_start = usize::MAX;
            ri.token_idx_stop = 0;
        }
        let mut last_lexeme = 0;
        let (indices, bytes) = self.get_bytes_and_lexeme_indices();
        let mut byte_idx = 0;

        for t in tokens {
            for b in trie.token(*t).iter() {
                if num_skip > 0 {
                    num_skip -= 1;
                    continue;
                }

                if byte_idx >= bytes.len() {
                    self.token_idx = tok_idx; // save local pointer, in case push_row() uses it
                    if !self.try_push_byte_definitive(*b) {
                        return Ok("parse reject");
                    }

                    let item_count = self.curr_row().item_indices().count();
                    if item_count > MAX_ROW {
                        bail!(
                            "Current row has {} items; max is {}; consider making your grammar left-recursive if it's right-recursive",
                            item_count,
                            MAX_ROW,
                        );
                    }
                    last_lexeme = self.num_rows() - 1;
                } else {
                    loop {
                        self.row_infos[last_lexeme].apply_token_idx(tok_idx);
                        if last_lexeme >= indices[byte_idx] {
                            break;
                        }
                        last_lexeme += 1;
                    }

                    if bytes[byte_idx] != *b {
                        println!(
                            "byte mismatch: {} != {} at {}",
                            bytes[byte_idx], b, last_lexeme
                        );
                        return Ok("static reject");
                    }
                }

                byte_idx += 1;
            }
            tok_idx += 1;
        }
        while last_lexeme < self.row_infos.len() {
            self.row_infos[last_lexeme].apply_token_idx(tok_idx);
            last_lexeme += 1;
        }
        self.token_idx = tok_idx;

        // self.print_row(self.num_rows() - 1);

        return Ok("");
    }

    pub fn filter_max_tokens(&mut self) {
        self.assert_definitive();

        let mut dst = 0;

        self.row_infos.push(RowInfo {
            lexeme: Lexeme::bogus(),
            token_idx_start: self.token_idx,
            token_idx_stop: self.token_idx,
            max_tokens: usize::MAX,
        });

        for idx in 0..self.num_rows() {
            let range = self.rows[idx].item_indices();
            self.rows[idx].first_item = dst;
            for i in range {
                let item = self.scratch.items[i];
                let item_props = &self.scratch.item_props[i];
                let sym_data = self.item_sym_data(&item);
                let max_tokens = sym_data.props.max_tokens;
                if max_tokens != usize::MAX {
                    let start_token_idx = self.row_infos[item.start_pos() + 1].token_idx_start;
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
        trace!("force_bytes lexer_stack {}", self.lexer_stack.len());
        let mut bytes = vec![];
        while let Some(b) = self.forced_byte() {
            debug!("  forced: {:?}", b as char);
            if !self.try_push_byte_definitive(b) {
                // shouldn't happen?
                info!("  force_bytes reject {}", b as char);
                break;
            }
            bytes.push(b);
        }
        trace!(
            "force_bytes exit {} lexer_stack={}",
            bytes.len(),
            self.lexer_stack.len()
        );
        bytes
    }

    #[inline(always)]
    fn advance_lexer_or_parser(
        &mut self,
        byte: u8,
        curr: LexerState,
        next_state: StateID,
        lexeme: Option<LexemeIdx>,
    ) -> bool {
        match lexeme {
            None => {
                // lexer advanced, but no lexeme
                self.lexer_stack.push(LexerState {
                    row_idx: curr.row_idx,
                    lexer_state: next_state,
                    byte,
                    use_byte: true,
                });
                true
            }
            Some(lexeme_idx) => self.advance_parser(next_state, lexeme_idx, Some(byte)),
        }
    }

    fn try_push_byte_definitive(&mut self, byte: u8) -> bool {
        assert!(self.scratch.definitive);

        let curr = self.lexer_state();
        let row = &self.rows[curr.row_idx as usize];

        debug!("B: {:?}", byte as char);
        let info = &self.row_infos[curr.row_idx as usize];
        let max_tokens_reached = info.token_idx_stop - info.token_idx_start > info.max_tokens;

        let res = if max_tokens_reached {
            // TODO the byte gets lost
            let (state, lexeme) =
                self.lexer
                    .force_lexeme_end(&row.allowed_lexemes, curr.lexer_state, Some(byte));
            Some((state, lexeme))
        } else {
            self.lexer.advance(
                &row.allowed_lexemes,
                curr.lexer_state,
                byte,
                self.scratch.definitive,
            )
        };

        if let Some((next_state, lexeme)) = res {
            self.advance_lexer_or_parser(byte, curr, next_state, lexeme)
        } else {
            debug!(
                "  lexer fail; allowed: {}",
                self.lexer_spec().dbg_lexeme_set(&row.allowed_lexemes)
            );
            false
        }
    }

    fn curr_row(&self) -> &Row {
        &self.rows[self.lexer_state().row_idx as usize]
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
        self.trie_started();
        for b in 0..=255 {
            if self.try_push_byte(b) {
                self.pop_bytes(1);
                if byte_sym.is_some() {
                    self.trie_finished();
                    return None; // more than one option
                } else {
                    byte_sym = Some(b);
                }
            }
        }
        self.trie_finished();
        byte_sym
    }

    fn flush_lexer(&mut self) -> bool {
        let curr = self.lexer_state();
        let row = self.curr_row();
        let (next_state, lexeme) =
            self.lexer
                .force_lexeme_end(&row.allowed_lexemes, curr.lexer_state, None);

        if let Some(lexeme) = lexeme {
            if !self.advance_parser(next_state, lexeme, None) {
                return false;
            }
        }

        true
    }

    pub fn scan_model_variable(&mut self, mv: ModelVariable) -> bool {
        self.assert_definitive(); // ???

        debug!("  scan mv: {:?}", mv);

        if !self.flush_lexer() {
            debug!("  flush_lexer() failed");
            return false;
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
            debug!("  scan_model_variable: no items");
            false
        } else {
            let r = self.push_row(self.scratch.row_start, &Lexeme::bogus());
            debug!("  scan_model_variable: {}", r);
            r
        }
    }

    // lexeme body only used for captures (in definitive mode)
    // and debugging (lexeme.idx used always)
    fn scan(&mut self, lexeme: &Lexeme) -> bool {
        let row_idx = self.num_rows() - 1;
        let last = self.rows[row_idx].last_item;
        let mut i = self.rows[row_idx].first_item;
        let n = last - i;
        self.scratch.ensure_items(last + n + 100);
        self.scratch.new_row(last);

        let trg = self.grammar.lexeme_to_sym_idx(lexeme.idx);

        if self.scratch.definitive {
            debug!(
                "scan: {} at {}",
                self.lexer_spec().dbg_lexeme(&lexeme),
                row_idx
            );
        }

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

    // lexeme only used for captures (in definitive mode)
    #[inline(always)]
    fn push_row(&mut self, mut agenda_ptr: usize, lexeme: &Lexeme) -> bool {
        let curr_idx = self.num_rows();
        let mut allowed_lexemes = SimpleVob::alloc(self.grammar.num_terminals());
        let mut max_tokens = 0;

        self.stats.rows += 1;

        while agenda_ptr < self.scratch.row_end {
            let item_idx = agenda_ptr;
            let item = self.scratch.items[agenda_ptr];
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
            } else {
                let sym_data = self.grammar.sym_data(after_dot);
                if let Some(lx) = self.grammar.lexeme_idx_of(after_dot) {
                    allowed_lexemes.set(lx.0, true);
                    max_tokens = max_tokens.max(sym_data.props.max_tokens);
                }
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

            if self.scratch.definitive {
                debug!(
                    "  push row: {}",
                    self.lexer_spec().dbg_lexeme_set(&allowed_lexemes)
                );
            }

            let idx = self.num_rows();
            let row = self.scratch.work_row(allowed_lexemes);
            if self.lexer_stack.len() == 1 || self.rows.len() == idx {
                self.rows.push(row);
            } else {
                self.rows[idx] = row;
            }

            if self.scratch.definitive {
                if self.row_infos.len() > idx {
                    self.row_infos.drain(idx..);
                }
                self.row_infos.push(RowInfo {
                    lexeme: Lexeme::bogus(),
                    token_idx_start: self.token_idx,
                    token_idx_stop: self.token_idx,
                    max_tokens,
                });
                debug!("  push: {idx} {} {}", self.rows.len(), self.row_infos.len());
            }

            true
        }
    }

    #[inline(always)]
    fn curr_row_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];
        let row_idx = self.num_rows() - 1;
        for back in self.lexer_stack.iter().rev() {
            if back.row_idx as usize != row_idx {
                break;
            }
            if back.use_byte {
                bytes.push(back.byte);
            }
        }
        bytes.reverse();
        bytes
    }

    fn lexer_spec(&self) -> &LexerSpec {
        self.grammar.lexer_spec()
    }

    #[inline(always)]
    fn mk_lexeme(&self, byte: Option<u8>, lexeme_idx: LexemeIdx) -> Lexeme {
        let mut bytes = self.curr_row_bytes();
        if byte.is_some() && !self.lexer_spec().greedy {
            bytes.push(byte.unwrap());
        }
        self.lexer_spec()
            .new_lexeme(lexeme_idx, bytes)
            .expect("TODO max_token, no stop token?")
    }

    fn has_forced_bytes(&self, allowed_lexemes: &SimpleVob, bytes: &[u8]) -> bool {
        for lexeme_idx in allowed_lexemes.iter() {
            let lex_spec = &self.lexer_spec().lexemes[lexeme_idx as usize];
            if !lex_spec.has_forced_bytes(bytes) {
                return false;
            }
        }
        true
    }

    /// Advance the parser with given lexeme_idx.
    /// lexer_state is state *after* consuming the byte.
    /// It either initial lexer states for lazy lexers,
    /// or lexer_initial_state+byte for greedy lexers.
    /// lexer_byte is the byte that led to producing the lexeme.
    #[inline(always)]
    fn advance_parser(
        &mut self,
        lexer_state: StateID,
        lexeme_idx: LexemeIdx,
        lexer_byte: Option<u8>,
    ) -> bool {
        let lexeme = if self.scratch.definitive {
            let res = self.mk_lexeme(lexer_byte, lexeme_idx);
            debug!("  lexer -> {}", self.lexer_spec().dbg_lexeme(&res));
            res
        } else {
            Lexeme::just_idx(lexeme_idx)
        };
        if self.scan(&lexeme) {
            let added_row = self.num_rows();
            let row_idx = added_row as u32;
            if self.scratch.definitive {
                // save lexeme at the new row, before we mess with the stack
                self.row_infos[added_row - 1].lexeme = lexeme;
            }
            let lex_spec = self.lexer_spec().lexeme_spec(lexeme_idx);
            if self.scratch.definitive {
                debug!("  lexeme: {:?}", lex_spec);
            }
            let no_hidden = LexerState {
                row_idx,
                lexer_state,
                byte: lexer_byte.unwrap_or(0),
                use_byte: lexer_byte.is_some() && self.lexer_spec().greedy,
            };
            if lexer_byte.is_some() && lex_spec.has_hidden_len() {
                // greedy lexers don't have stop tokens
                assert!(!self.lexer_spec().greedy);

                // make sure we have a real lexeme
                let lexeme = self.mk_lexeme(lexer_byte, lexeme_idx);

                let hidden_bytes = lexeme.hidden_bytes();
                let allowed_lexemes = &self.rows[added_row].allowed_lexemes;

                if self.scratch.definitive {
                    trace!(
                        "  allowed lexemes: {:?}",
                        self.lexer_spec().dbg_lexeme_set(allowed_lexemes)
                    );
                    trace!("  hidden: {:?}", String::from_utf8_lossy(&hidden_bytes));
                }

                if hidden_bytes.len() == 0 {
                    self.lexer_stack.push(no_hidden);
                } else if self.has_forced_bytes(allowed_lexemes, &hidden_bytes) {
                    if self.scratch.definitive {
                        trace!("  hidden forced");
                    }
                    // if the bytes are forced, we just advance the lexer
                    // by replacing the top lexer states
                    self.pop_lexer_states(hidden_bytes.len() - 1);
                    let allowed_lexemes = &self.rows[added_row].allowed_lexemes;
                    let mut lexer_state = lexer_state;
                    for b in hidden_bytes {
                        match self.lexer.advance(
                            allowed_lexemes,
                            lexer_state,
                            *b,
                            self.scratch.definitive,
                        ) {
                            Some((next_state, None)) => {
                                lexer_state = next_state;
                            }
                            None => panic!("hidden byte failed"),
                            _ => panic!("hidden byte produced lexeme"),
                        }
                        self.lexer_stack.push(LexerState {
                            row_idx,
                            lexer_state,
                            byte: *b,
                            use_byte: true,
                        });
                    }
                } else {
                    // prevent any further matches in this branch
                    self.lexer_stack.push(LexerState {
                        lexer_state: self.lexer.a_dead_state(),
                        use_byte: false, // ?
                        ..no_hidden
                    });
                }
            } else {
                self.lexer_stack.push(no_hidden);
            }
            if self.scratch.definitive {
                self.assert_definitive();
            }
            true
        } else {
            if self.scratch.definitive {
                debug!("  scan failed");
            }
            false
        }
    }
}

impl Recognizer for Parser {
    fn pop_bytes(&mut self, num: usize) {
        self.pop_lexer_states(num);
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
        // debug!("trie_started: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        self.assert_definitive();
        self.trie_lexer_stack = self.lexer_stack.len();
        self.scratch.definitive = false;
    }

    fn trie_finished(&mut self) {
        // debug!("trie_finished: rows={} lexer={}", self.num_rows(), self.lexer_stack.len());
        assert!(self.scratch.definitive == false);
        assert!(self.row_infos.len() <= self.num_rows());
        // clean up stack
        self.pop_lexer_states(self.lexer_stack.len() - self.trie_lexer_stack);
        self.scratch.definitive = true;
        self.assert_definitive();
    }

    #[inline(always)]
    fn try_push_byte(&mut self, byte: u8) -> bool {
        assert!(!self.scratch.definitive);

        let lexer_logging = false;
        let curr = self.lexer_state();
        let res = self.lexer.advance(
            &self.rows[curr.row_idx as usize].allowed_lexemes,
            curr.lexer_state,
            byte,
            lexer_logging,
        );

        if let Some((next_state, lexeme)) = res {
            self.advance_lexer_or_parser(byte, curr, next_state, lexeme)
        } else {
            false
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
