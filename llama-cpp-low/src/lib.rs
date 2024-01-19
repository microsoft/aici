#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use anyhow::{anyhow, Result};
use core::slice;
use std::ffi::CString;

extern crate link_cplusplus;

#[allow(dead_code)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use sys::*;

pub struct Model {
    model: *mut llama_model,
    ctx: *mut llama_context,
}

pub struct Batch {
    size: usize,
    batch: llama_batch,
}

pub type ModelParams = llama_model_params;
pub type ContextParams = llama_context_params;

impl Default for ModelParams {
    fn default() -> Self {
        unsafe { llama_model_default_params() }
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

    pub fn enable_logits(&mut self, tok_idx: usize) {
        let p = tok_idx as usize;
        assert!(p < self.size);
        unsafe {
            self.batch.logits.add(p).write(1);
        }
    }

    pub fn add_token(&mut self, token: u32, pos: usize, seq_id: i32, logits: bool) {
        let p = self.batch.n_tokens as usize;
        assert!(p < self.size);
        unsafe {
            self.batch.token.add(p).write(token as i32);
            self.batch.pos.add(p).write(pos as i32);
            self.batch.n_seq_id.add(p).write(seq_id);
            self.batch.seq_id.add(p).read().add(0).write(seq_id);
            self.batch.logits.add(p).write(if logits { 1 } else { 0 });
        }
        self.batch.n_tokens += 1;
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.batch);
        }
    }
}

impl Model {
    pub fn from_file(file: &str, mparams: ModelParams, cparams: ContextParams) -> Self {
        unsafe {
            let numa = false;
            llama_backend_init(numa); // TODO: only call this once? also numa?
            let c = CString::new(file).unwrap();
            let model = llama_load_model_from_file(c.as_ptr(), mparams);
            assert!(model != std::ptr::null_mut());
            let ctx = llama_new_context_with_model(model, cparams);
            assert!(ctx != std::ptr::null_mut());
            Model { model, ctx }
        }
    }

    pub fn vocab_size(&self) -> usize {
        unsafe { llama_n_vocab(self.model) as usize }
    }

    pub fn token_to_bytes(&mut self, token: u32) -> Vec<u8> {
        let mut sz = 32;
        loop {
            let mut res = vec![0u8; sz];
            let ntok = unsafe {
                llama_token_to_piece(
                    self.model,
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
    pub fn tokenize(&mut self, data: &[u8], add_bos: bool, special: bool) -> Vec<u32> {
        let mut res = vec![0u32; data.len()];
        let ntok = unsafe {
            llama_tokenize(
                self.model,
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

    pub fn decode(&mut self, batch: &mut Batch) -> Result<()> {
        let r = unsafe { llama_decode(self.ctx, batch.batch) };
        if r == 1 {
            Err(anyhow!("KV cache overflow"))
        } else if r != 0 {
            Err(anyhow!("decode failed: {r}"))
        } else {
            Ok(())
        }
    }

    pub fn get_logits(&mut self, idx: usize) -> &[f32] {
        let n = self.vocab_size();
        unsafe { slice::from_raw_parts(llama_get_logits_ith(self.ctx, idx as i32), n) }
    }
}
