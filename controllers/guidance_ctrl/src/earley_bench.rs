fn main() {
    #[cfg(not(target_os = "wasi"))]
    bench();
}

#[cfg(not(target_os = "wasi"))]
fn bench() {
    use aici_abi::toktree::TokTrie;
    use aici_guidance_ctrl::earley::bench::earley_test;
    earley_test(TokTrie::from_bytes(&tokenizer::token_trie_bytes()));
}
