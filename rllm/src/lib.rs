pub mod attn;
mod kernels;
pub mod llama;
mod logits;
pub mod phi;
mod playground;
pub mod refkernels;
pub mod seq;

// vllm modules
mod blocks;
mod cache_engine;
pub mod config;
mod engine;
pub mod iface;
mod scheduler;
pub mod util;

use std::sync::atomic::AtomicBool;

pub use engine::*;
pub use logits::LogitsProcessor;
pub use playground::playground_1;

pub use tch::Kind as DType;
pub use tch::{Device, IndexOp, Shape, Tensor};

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
        let (device, dtype) = if tch::Cuda::is_available() {
            (Device::Cuda(0), DType::BFloat16)
        } else {
            // At least on AMD 5500m MPS is 3x slower than CPU
            // #[cfg(target_os = "macos")]
            // let r = (Device::Mps, DType::Half);
            // #[cfg(not(target_os = "macos"))]
            let r = (Device::Cpu, DType::Float);
            r
        };
        Self {
            tokenizer: "llama".to_string(),
            model_id: "NousResearch/Llama-2-7b-hf".to_string(),
            revision: None,
            local_weights: None,
            alt: 0,
            dtype,
            device,
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
