mod scheduler;

pub use scheduler::*;

cfg_if::cfg_if! {
    if #[cfg(not(feature = "llamacpp"))] {
        #[cfg(not(feature = "cuda"))]
        mod cuda_stub;

        mod blocks;
        mod cache_engine;
        mod batch_info;

        pub use batch_info::*;
        pub use cache_engine::*;
        pub use blocks::*;
    }
}

pub struct CacheSize {
    pub gpu: usize,
    pub cpu: usize,
}
