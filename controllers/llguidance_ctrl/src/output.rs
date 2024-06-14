use aici_abi::bytes::to_hex_string;
use serde::{Deserialize, Serialize};

use crate::{earley, TokenParser};

#[derive(Serialize, Deserialize)]
#[serde(tag = "object", rename_all = "snake_case")]
pub enum ParserOutput {
    Capture {
        name: String,
        str: String,
        hex: String,
        log_prob: f64,
    },
    FinalText {
        str: String,
        hex: String,
    },
    Text {
        str: String,
        hex: String,
        log_prob: f64,
        num_tokens: usize,
    },
    Stats {
        #[serde(flatten)]
        stats: earley::ParserStats,
    },
}

impl ParserOutput {
    pub fn text_from_bytes(bytes: &[u8], log_prob: f64, num_tokens: usize) -> Self {
        ParserOutput::Text {
            str: String::from_utf8_lossy(bytes).to_string(),
            hex: to_hex_string(bytes),
            log_prob,
            num_tokens,
        }
    }

    pub fn final_text_from_bytes(bytes: &[u8]) -> Self {
        ParserOutput::FinalText {
            str: String::from_utf8_lossy(bytes).to_string(),
            hex: to_hex_string(bytes),
        }
    }
}

pub struct Reporter {
    reported_captures: usize,
    text_ptr: usize,
    token_ptr: usize,
    prev_stats: earley::ParserStats,
}

impl Reporter {
    pub fn new(tok_parser: &TokenParser) -> Self {
        Reporter {
            reported_captures: 0,
            text_ptr: 0,
            token_ptr: tok_parser.num_tokens(),
            prev_stats: tok_parser.parser.stats().clone(),
        }
    }

    pub fn get_progress(
        &mut self,
        tok_parser: &mut TokenParser,
        is_final: bool,
    ) -> Vec<ParserOutput> {
        let mut res = vec![];
        // first report newly generated text
        let num_tokens = tok_parser.num_tokens();
        let new_text = tok_parser.bytes_since(self.text_ptr);
        if new_text.len() > 0 {
            // TODO log_prob
            let text = ParserOutput::text_from_bytes(new_text, 0.0, num_tokens - self.token_ptr);
            res.push(text);
            self.text_ptr += new_text.len();
            self.token_ptr = num_tokens;
        }

        // then the captures
        let captures = &tok_parser.parser.captures()[self.reported_captures..];
        self.reported_captures += captures.len();

        // remove duplicate names
        let mut seen = std::collections::HashSet::new();
        let captures = captures
            .iter()
            .rev()
            .filter(|(name, _)| seen.insert(name))
            .collect::<Vec<_>>();
        for (name, val) in captures.iter().rev() {
            let cap = ParserOutput::Capture {
                name: name.clone(),
                str: String::from_utf8_lossy(val).to_string(),
                hex: to_hex_string(val),
                log_prob: 0.0, // TODO
            };
            res.push(cap);
        }

        if is_final {
            let final_text = ParserOutput::final_text_from_bytes(tok_parser.final_bytes());
            res.push(final_text);
        }

        let delta = tok_parser.parser.stats().delta(&self.prev_stats);
        self.prev_stats = tok_parser.parser.stats().clone();
        res.push(ParserOutput::Stats { stats: delta });

        res
    }
}
