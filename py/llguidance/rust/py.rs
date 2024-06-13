use std::{borrow::Cow, sync::Arc};

use aici_abi::{
    bytes::TokRxInfo,
    toktree::{self, TokTrie},
    MidProcessArg, TokenId, TokenizerEnv,
};
use aici_llguidance_ctrl::{api::TopLevelGrammar, output::Reporter, TokenParser};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};

#[pyclass]
struct LLInterpreter {
    inner: TokenParser,
    temperature: f32,
    reporter: Reporter,
    #[pyo3(get, set)]
    log_level: isize,
}

#[pyclass]
struct LLTokenizer {
    tok_trie: Arc<toktree::TokTrie>,
}

#[pymethods]
impl LLInterpreter {
    #[new]
    fn py_new(
        tokenizer: &LLTokenizer,
        llguidance_json: &str,
        log_level: Option<isize>,
    ) -> PyResult<Self> {
        let env = PyTokenizer {
            inner: tokenizer.tok_trie.clone(),
        };
        let arg: TopLevelGrammar = serde_json::from_str(llguidance_json)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let log_level = log_level.unwrap_or(1);
        let inner = TokenParser::from_llguidance_json(Box::new(env), arg, log_level)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reporter = Reporter::new(&inner);
        Ok(LLInterpreter {
            inner,
            reporter,
            temperature: 0.0,
            log_level,
        })
    }

    fn process_prompt(&mut self, prompt: Vec<TokenId>) -> Vec<TokenId> {
        self.inner.process_prompt(prompt)
    }

    fn mid_process(&mut self, backtrack: u32, tokens: Vec<TokenId>) -> (Option<Cow<[u8]>>, String) {
        let r = self.inner.mid_process(MidProcessArg {
            backtrack,
            tokens,
            fork_group: vec![],
        });
        let is_final = r.is_stop();
        let mut res = PyMidProcessResult {
            progress: self.reporter.get_progress(&mut self.inner, is_final),
            stop: is_final,
            backtrack: 0,
            temperature: self.temperature,
            ff_tokens: vec![],
        };
        if is_final {
            (None, serde_json::to_string(&res).unwrap())
        } else {
            assert!(r.branches.len() == 1);
            let b = &r.branches[0];
            if b.temperature.is_some() {
                self.temperature = b.temperature.unwrap();
                res.temperature = self.temperature;
            }
            if b.splices.len() > 0 {
                assert!(b.splices.len() == 1);
                assert!(b.splices[0].when_sampled.is_empty());
                res.backtrack = b.splices[0].backtrack;
                res.ff_tokens = b.splices[0].ff_tokens.clone();
            }
            let mask = b.sample_mask.as_ref().map(|m| {
                let mut res = vec![0u8; m.len()];
                m.iter_set_entries(|i| res[i] = 200);
                res.pop();
                Cow::Owned(res)
            });
            (mask, serde_json::to_string(&res).unwrap())
        }
    }
}

#[derive(Serialize, Deserialize)]
struct PyMidProcessResult {
    progress: Vec<serde_json::Value>,
    stop: bool,
    backtrack: u32,
    ff_tokens: Vec<TokenId>,
    temperature: f32,
}

#[pymethods]
impl LLTokenizer {
    #[new]
    fn py_new(eos_token: u32, tokens: Vec<Vec<u8>>) -> PyResult<Self> {
        let info = TokRxInfo {
            vocab_size: tokens.len() as u32,
            tok_eos: eos_token,
        };

        let tok_trie = TokTrie::from(&info, &tokens);
        Ok(LLTokenizer {
            tok_trie: Arc::new(tok_trie),
        })
    }

    fn greedy_tokenize(&self, text: &str) -> Vec<u32> {
        self.tok_trie.greedy_tokenize(text.as_bytes())
    }

    fn dbg_tokens(&self, tokens: Vec<u32>) -> String {
        self.tok_trie.tokens_dbg(&tokens)
    }

    fn decode_str(&self, tokens: Vec<u32>) -> String {
        self.tok_trie.decode_str(&tokens)
    }

    fn decode_bytes(&self, tokens: Vec<u32>) -> Cow<[u8]> {
        let r = self.tok_trie.decode(&tokens);
        Cow::Owned(r)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.tok_trie.vocab_size() as usize
    }

    #[getter]
    fn eos_token(&self) -> u32 {
        self.tok_trie.eos_token()
    }
}

struct PyTokenizer {
    inner: Arc<toktree::TokTrie>,
}

impl TokenizerEnv for PyTokenizer {
    fn stop(&self) -> ! {
        panic!("STOP"); // TODO
    }

    fn tok_trie(&self) -> &toktree::TokTrie {
        &self.inner
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        // TODO this should call out to the Python tokenizer
        self.inner.greedy_tokenize(s)
    }
}

pub(crate) fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LLTokenizer>()?;
    m.add_class::<LLInterpreter>()?;
    Ok(())
}
