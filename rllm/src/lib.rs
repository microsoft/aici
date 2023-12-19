pub mod seq;
pub mod llm;
pub mod paged;

// vllm modules
pub mod config;
mod engine;
pub mod iface;
pub mod util;

use std::sync::atomic::AtomicBool;

use config::AiciConfig;
pub use engine::*;
pub use llm::logits::LogitsProcessor;

pub use tch::Kind as DType;
pub use tch::{Device, IndexOp, Shape, Tensor};

pub struct LoaderArgs {
    pub tokenizer: String, // one of aici_tokenizer; eg "llama"
    pub model_id: String,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
    pub alt: usize,
    pub aici: AiciConfig,

    pub dtype: Option<DType>,
    pub device: Device,
}

impl Default for LoaderArgs {
    fn default() -> Self {
        let (device, dtype) = if tch::Cuda::is_available() {
            (Device::Cuda(0), None)
        } else {
            // At least on AMD 5500m MPS is 3x slower than CPU
            // #[cfg(target_os = "macos")]
            // let r = (Device::Mps, DType::Half);
            // #[cfg(not(target_os = "macos"))]
            let r = (Device::Cpu, Some(DType::Float));
            r
        };
        Self {
            tokenizer: "llama".to_string(),
            model_id: "NousResearch/Llama-2-7b-hf".to_string(),
            revision: None,
            local_weights: None,
            aici: AiciConfig::default(),
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
