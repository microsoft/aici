use aici_abi::{bytes::to_hex_string, MidProcessResult};
use serde::{Deserialize, Serialize};

use crate::{earley, TokenParser};

#[derive(Serialize, Deserialize)]
pub struct BytesOutput {
    pub str: String,
    pub hex: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "object", rename_all = "snake_case")]
pub enum ParserOutput {
    Capture {
        name: String,
        #[serde(flatten)]
        bytes: BytesOutput,
        log_prob: f64,
    },
    FinalText {
        #[serde(flatten)]
        bytes: BytesOutput,
    },
    Text {
        #[serde(flatten)]
        bytes: BytesOutput,
        log_prob: f64,
        num_tokens: usize,
        is_generated: bool,
        stats: ParserStats,
    },
}

#[derive(Serialize, Deserialize)]
pub struct ParserStats {
    runtime_us: u64,
    #[serde(flatten)]
    stats: earley::ParserStats,
}

impl From<&[u8]> for BytesOutput {
    fn from(bytes: &[u8]) -> Self {
        BytesOutput::from_bytes(bytes)
    }
}

impl BytesOutput {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        BytesOutput {
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
    is_generated: bool,
}

impl Reporter {
    pub fn new(tok_parser: &TokenParser) -> Self {
        Reporter {
            reported_captures: 0,
            text_ptr: 0,
            token_ptr: tok_parser.num_tokens(),
            prev_stats: tok_parser.parser_stats().clone(),
            is_generated: false,
        }
    }

    pub fn get_progress(
        &mut self,
        tok_parser: &mut TokenParser,
        mid_res: &MidProcessResult,
    ) -> Vec<ParserOutput> {
        let mut res = vec![];

        // start with captures
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
            res.push(ParserOutput::Capture {
                name: name.clone(),
                bytes: val.as_slice().into(),
                log_prob: 0.0, // TODO
            });
        }

        // compute stats
        let delta = tok_parser.parser_stats().delta(&self.prev_stats);
        self.prev_stats = tok_parser.parser_stats().clone();
        let runtime_us = tok_parser.mid_process_start_time.elapsed().as_micros() as u64;
        let stats = ParserStats {
            runtime_us,
            stats: delta,
        };

        // report newly generated text
        let num_tokens = tok_parser.num_tokens();
        let new_text = tok_parser.bytes_since(self.text_ptr);
        res.push(ParserOutput::Text {
            bytes: new_text.into(),
            log_prob: 0.0, // TODO
            num_tokens: num_tokens - self.token_ptr,
            is_generated: self.is_generated,
            stats,
        });
        self.text_ptr += new_text.len();
        self.token_ptr = num_tokens;

        self.is_generated = mid_res.branches.len() >= 1 && mid_res.branches[0].splices.len() == 0;

        if mid_res.is_stop() {
            res.push(ParserOutput::FinalText {
                bytes: tok_parser.final_bytes().into(),
            });
        }

        res
    }
}
