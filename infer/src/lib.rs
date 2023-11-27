mod kernels;
pub mod llama;
mod logits;
mod playground;
pub mod seq;

// vllm modules
mod blocks;
mod cache_engine;
pub mod config;
mod engine;
mod scheduler;

use std::sync::atomic::AtomicBool;

pub use engine::RllmEngine;
pub use kernels::*;
pub use logits::LogitsProcessor;
pub use playground::playground_1;

#[derive(Default)]
pub struct LoaderArgs {
    pub model_id: Option<String>,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
    pub use_reference: bool,
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
