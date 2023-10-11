use aici_abi::{
    aici_expose_all, include_bytes_aligned,
    rx::{StateOffset, TokRx},
    rxvm::RxAici,
    AiciVm,
};

fn tokrx() -> TokRx {
    TokRx::deserialize(include_bytes_aligned!(u32, "../../regex_llm/rx.bin"))
}

aici_expose_all!(RxAici, RxAici::from_token_compiled(tokrx()));

fn main() {
    let t = tokrx();
    let mut state = StateOffset::START;
    state = t.advance(state, 1);
    state = t.advance(state, 2);
    _ = state;
    println!("Deserialization test OK")
}
