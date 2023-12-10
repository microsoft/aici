mod kernels;
pub mod llama;
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

pub use candle_core::{Device, Tensor, DType, IndexOp, Shape, D};

#[derive(Default)]
pub struct LoaderArgs {
    pub tokenizer: String, // one of aici_tokenizer; eg "llama"
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
    pub alt: usize,
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
