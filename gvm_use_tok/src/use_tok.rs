use gvm_abi::toktree::{walk, TokTrie};

fn main() {
    let trie = TokTrie::from_bytes(include_bytes!("tokenizers/gpt4.bin"));
    for idx in 1000..1001 {
        let bytes = trie.token(idx);
        println!("{}: {:?}", idx, String::from_utf8_lossy(bytes));
    }

    let max_len = (0..trie.info().vocab_size).map(|idx| trie.token(idx).len()).max();
    println!("max_len: {}", max_len.unwrap());

    let mut sum = 0;
    for _ in 0..1000 {
        sum += walk(&trie);
    }
    println!("sum: {}", sum);
}
