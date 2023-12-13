mod kernels;
pub mod llama;
// pub mod phi;
pub mod attn;
mod logits;
mod playground;
pub mod seq;

// vllm modules
mod blocks;
mod cache_engine;
pub mod config;
pub mod util;
mod engine;
mod scheduler;
pub mod iface;

use std::sync::atomic::AtomicBool;

pub use engine::RllmEngine;
pub use engine::AddRequest;
pub use kernels::*;
pub use logits::LogitsProcessor;
pub use playground::playground_1;

pub use tch::{Device, Tensor, IndexOp, Shape};
pub use tch::Kind as DType;

pub struct LoaderArgs {
    pub tokenizer: String, // one of aici_tokenizer; eg "llama"
    pub model_id: String,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
    pub alt: usize,

    pub dtype: DType,
    pub device: Device,
}

impl Default for LoaderArgs {
    fn default() -> Self {
        Self {
            tokenizer: "llama".to_string(),
            model_id: "NousResearch/Llama-2-7b-hf".to_string(),
            revision: None,
            local_weights: None,
            alt: 0,
            dtype: DType::BFloat16,
            device: Device::Cuda(0),
        }
    }
}

static mut TRACE: AtomicBool = AtomicBool::new(false);

pub fn set_trace(trace_enabled: bool) {
    unsafe {
        TRACE.store(trace_enabled, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn get_trace() -> bool {
    unsafe { TRACE.load(std::sync::atomic::Ordering::Relaxed) }
}

#[macro_export]
macro_rules! rtrace {
    ($($arg:tt)*) => {{
        if $crate::get_trace() {
            println!($($arg)*);
        }
    }};
}
