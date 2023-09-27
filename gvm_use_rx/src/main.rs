use gvm_abi::{
    gvm_expose_all, include_bytes_aligned, println,
    rx::{StateOffset, TokRx},
    rxvm::RxGvm,
    GuidanceVm,
};

fn tokrx() -> TokRx {
    TokRx::deserialize(include_bytes_aligned!(u32, "../../regex_llm/rx.bin"))
}

gvm_expose_all!(RxGvm, RxGvm::from_token_compiled(tokrx()));

fn main() {
    let t = tokrx();
    let mut state = StateOffset::START;
    state = t.advance(state, 1);
    state = t.advance(state, 2);
    _ = state;
    println!("Deserialization test OK")
}
