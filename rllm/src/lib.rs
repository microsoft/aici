pub mod paged;
pub mod seq;

// vllm modules
pub mod config;
mod engine;
pub mod iface;
mod logits;
pub mod util;

use config::AiciConfig;
use config::CommonModelConfig;
use config::ModelMeta;
pub use engine::*;
pub use logits::LogitsProcessor;
use std::sync::atomic::AtomicBool;

cfg_if::cfg_if! {
    if #[cfg(feature = "tch")] {
        pub mod llm;
        pub use tch::{Device, IndexOp, Kind as DType, Shape, Tensor};
        pub(crate) use paged::BlockRef;
        pub(crate) use paged::BlockSpaceManager;
    } else {
        pub mod llamacpp;
        pub use llamacpp as llm;
        pub use llm::{Device, DType, Tensor};
        pub(crate) use llamacpp::BlockRef;
        pub(crate) use llamacpp::blocks::BlockSpaceManager;
    }
}

pub use llm::{tmodel::TModel, util::to_vec1};

pub use fxhash::FxHashMap as HashMap;
pub use fxhash::FxHashSet as HashSet;

pub struct LoaderArgs {
    pub tokenizer: String, // one of aici_tokenizer; eg "llama"
    pub model_id: String,
    pub revision: Option<String>,
    pub local_weights: Option<String>,
    pub gguf: Option<String>,
    pub alt: usize,
    pub aici: AiciConfig,

    #[cfg(not(feature = "tch"))]
    pub(crate) cached_model: Option<llamacpp::Model>,

    pub dtype: Option<DType>,
    pub device: Device,
}

impl Default for LoaderArgs {
    fn default() -> Self {
        #[cfg(feature = "tch")]
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
        #[cfg(not(feature = "tch"))]
        let (device, dtype) = (Device::Cpu, Some(DType::Float));
        Self {
            tokenizer: "llama".to_string(),
            model_id: "NousResearch/Llama-2-7b-hf".to_string(),
            revision: None,
            local_weights: None,
            gguf: None,
            aici: AiciConfig::default(),
            alt: 0,
            dtype,
            device,
            #[cfg(not(feature = "tch"))]
            cached_model: None,
        }
    }
}

impl LoaderArgs {
    pub fn common_config(&self) -> CommonModelConfig {
        CommonModelConfig {
            meta: ModelMeta {
                id: self.model_id.clone(),
            },
            dtype: self.dtype,
            device: self.device.clone(),
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
