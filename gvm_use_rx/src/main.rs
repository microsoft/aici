use gvm_abi::{gvm_expose_all, include_bytes_aligned, rx::TokRx, rxvm::RxGvm, GuidanceVm};

gvm_expose_all!(
    RxGvm,
    RxGvm::from_token_compiled(TokRx::deserialize(include_bytes_aligned!(
        u32,
        "../../regex_llm/rx.bin"
    )))
);

fn main() {}
