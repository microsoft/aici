use anyhow::{anyhow, bail, Result};
use core::slice;
use std::{
    ffi::CString,
    fmt::{Debug, Formatter},
    sync::{Arc, Mutex},
};

extern crate link_cplusplus;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use sys::*;

struct ModelInner {
    seq_id: i32,
    model: *mut llama_model,
    ctx: *mut llama_context,
}

unsafe impl Send for ModelInner {}

#[derive(Clone)]
pub struct Model {
    inner: Arc<Mutex<ModelInner>>,
}

pub struct Batch {
    size: usize,
    batch: llama_batch,
}

pub struct Sequence {
    id: i32,
    model: Model,
}

pub type ModelParams = llama_model_params;
pub type ContextParams = llama_context_params;

impl Default for ModelParams {
    fn default() -> Self {
        unsafe { llama_model_default_params() }
    }
}

pub enum SplitMode {
    None = llama_split_mode_LLAMA_SPLIT_NONE as isize,
    /// split layers and KV across GPUs
    Layer = llama_split_mode_LLAMA_SPLIT_LAYER as isize,
    /// split rows across GPUs
    Row = llama_split_mode_LLAMA_SPLIT_ROW as isize,
}

impl ModelParams {
    pub fn set_split_mode(&mut self, mode: SplitMode) {
        self.split_mode = mode as u32;
    }
}

impl Default for ContextParams {
    fn default() -> Self {
        let mut r = unsafe { llama_context_default_params() };
        // get_num_physical_cores() in llama
        r.n_threads = std::cmp::min(num_cpus::get_physical() as u32, 32);
        r.n_threads_batch = r.n_threads;
        r
    }
}

impl Batch {
    /// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    /// This can either prompt or generation tokens.
    pub fn new(n_tokens: usize) -> Self {
        let batch = unsafe { llama_batch_init(n_tokens as i32, 0, 1) };
        Batch {
            size: n_tokens,
            batch,
        }
    }

    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }

    pub fn len(&self) -> usize {
        self.batch.n_tokens as usize
    }

    pub fn max_len(&self) -> usize {
        self.size
    }

    pub fn enable_logits(&mut self, tok_idx: usize) {
        let p = tok_idx as usize;
        assert!(p < self.size);
        unsafe {
            self.batch.logits.add(p).write(1);
        }
    }

    pub fn add_token(&mut self, token: u32, pos: usize, seq: &Sequence, logits: bool) {
        let p = self.batch.n_tokens as usize;
        assert!(p < self.size);
        unsafe {
            self.batch.token.add(p).write(token as i32);
            self.batch.pos.add(p).write(pos as i32);
            self.batch.n_seq_id.add(p).write(1);
            self.batch.seq_id.add(p).read().add(0).write(seq.id);
            self.batch.logits.add(p).write(if logits { 1 } else { 0 });
        }
        self.batch.n_tokens += 1;
    }
}

impl Debug for Batch {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Batch {{\n")?;
        write!(f, "  size: {}, n_tokens: {}\n", self.size, self.len())?;
        write!(f, "}}")
    }
}

pub struct ModelInfo {
    pub n_ctx_train: i32,
    pub n_embd: i32,
    pub n_vocab: i32,
    pub rope: f32,
}

impl Model {
    pub fn from_file(file: &str, mparams: ModelParams, cparams: ContextParams) -> Result<Self> {
        unsafe {
            let numa = false;
            llama_backend_init(numa); // TODO: only call this once?
            let c = CString::new(file).unwrap();
            let model = llama_load_model_from_file(c.as_ptr(), mparams);
            if model == std::ptr::null_mut() {
                bail!("failed to load model")
            }
            let ctx = llama_new_context_with_model(model, cparams);
            if ctx == std::ptr::null_mut() {
                bail!("failed to create context")
            }
            Ok(Model {
                inner: Arc::new(Mutex::new(ModelInner {
                    model,
                    ctx,
                    seq_id: 0,
                })),
            })
        }
    }

    pub fn new_sequence(&self) -> Sequence {
        let mut inner = self.inner.lock().unwrap();
        let seq_id = inner.seq_id;
        inner.seq_id += 1;
        Sequence {
            id: seq_id,
            model: self.clone(),
        }
    }

    pub fn model_info(&self) -> ModelInfo {
        unsafe {
            let model = self.inner.lock().unwrap().model;
            ModelInfo {
                n_ctx_train: llama_n_ctx_train(model),
                n_embd: llama_n_embd(model),
                n_vocab: llama_n_vocab(model),
                rope: llama_rope_freq_scale_train(model),
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        unsafe {
            let model = self.inner.lock().unwrap().model;
            llama_n_vocab(model) as usize
        }
    }

    pub fn token_to_bytes(&self, token: u32) -> Vec<u8> {
        let mut sz = 32;
        loop {
            let mut res = vec![0u8; sz];
            let ntok = unsafe {
                let model = self.inner.lock().unwrap().model;
                llama_token_to_piece(
                    model,
                    token as i32,
                    res.as_mut_ptr() as *mut i8,
                    res.len() as i32,
                )
            };
            if ntok < 0 {
                assert!(sz == 32);
                sz = -ntok as usize;
            } else {
                res.truncate(ntok as usize);
                return res;
            }
        }
    }

    /// Convert the provided text into tokens.
    /// `special` - Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.
    ///             Does not insert a leading space.
    pub fn tokenize(&self, data: &[u8], add_bos: bool, special: bool) -> Vec<u32> {
        let mut res = vec![0u32; data.len()];
        let ntok = unsafe {
            let model = self.inner.lock().unwrap().model;
            llama_tokenize(
                model,
                data.as_ptr() as *mut i8,
                data.len() as i32,
                res.as_mut_ptr() as *mut i32,
                res.len() as i32,
                add_bos,
                special,
            )
        };
        assert!(ntok >= 0);
        res.truncate(ntok as usize);
        res
    }

    pub fn decode(&self, batch: &mut Batch) -> Result<()> {
        let r = unsafe {
            let ctx = self.inner.lock().unwrap().ctx;
            llama_decode(ctx, batch.batch)
        };
        if r == 1 {
            Err(anyhow!("KV cache overflow"))
        } else if r != 0 {
            Err(anyhow!("decode failed: {r}"))
        } else {
            Ok(())
        }
    }

    pub fn get_logits(&self, idx: usize) -> &'static [f32] {
        let n = self.vocab_size();
        unsafe {
            let ctx = self.inner.lock().unwrap().ctx;
            slice::from_raw_parts(llama_get_logits_ith(ctx, idx as i32), n)
        }
    }
}

impl Sequence {
    pub fn id(&self) -> i32 {
        self.id
    }
    pub fn assert_model(&self, model: &Model) {
        assert!(Arc::ptr_eq(&self.model.inner, &model.inner));
    }
    pub fn rm(&self, start: i32, stop: i32) {
        unsafe {
            let ctx = self.model.inner.lock().unwrap().ctx;
            llama_kv_cache_seq_rm(ctx, self.id, start, stop);
        }
    }
    pub fn cp(&self, other: &Sequence, start: i32, stop: i32) {
        unsafe {
            let ctx = self.model.inner.lock().unwrap().ctx;
            llama_kv_cache_seq_cp(ctx, self.id, other.id, start, stop);
        }
    }
}

impl Drop for ModelInner {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ctx);
            llama_free_model(self.model);
        }
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        unsafe {
            let ctx = self.model.inner.lock().unwrap().ctx;
            llama_kv_cache_seq_rm(ctx, self.id, 0, -1);
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.batch);
        }
    }
}
