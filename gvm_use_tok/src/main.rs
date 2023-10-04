use gvm_abi::toktree::TokTrie;

fn main() {
    let trie = TokTrie::from_bytes(include_bytes!("tokenizers/gpt4.bin"));
    for idx in 1000..1001 {
        let bytes = trie.token(idx);
        println!("{}: {:?}", idx, String::from_utf8_lossy(bytes));
    }
    
}
