pub mod seq;

// vllm modules
pub mod config;
mod engine;
mod exec;
mod expected;
pub mod iface;
mod logits;
mod scheduler;
pub mod server;
pub mod util;

use config::AiciConfig;
pub use engine::*;
pub use exec::*;
pub use logits::LogitsProcessor;
pub use scheduler::*;
use std::sync::atomic::AtomicBool;

pub use aicirt::HashMap;
pub use aicirt::HashSet;

pub struct LoaderArgs {
    pub tokenizer: String, // one of aici_tokenizer; eg "llama"
    pub model_id: String,
    pub revision: Option<String>,
    pub file: Option<String>,
    pub local_weights: Option<String>,
    pub alt: usize,
    pub aici: AiciConfig,
}

impl Default for LoaderArgs {
    fn default() -> Self {
        Self {
            tokenizer: "llama".to_string(),
            model_id: "NousResearch/Llama-2-7b-hf".to_string(),
            revision: None,
            local_weights: None,
            file: None,
            aici: AiciConfig::default(),
            alt: 0,
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
