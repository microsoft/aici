use gvm_abi::{
    recognizer::{compute_bias, Recognizer, Uppercase},
    toktree::TokTrie,
};

fn main() {
    let trie = TokTrie::from_bytes(include_bytes!("tokenizers/gpt4.bin"));
    for idx in 1000..1001 {
        let bytes = trie.token(idx);
        println!("{}: {:?}", idx, String::from_utf8_lossy(bytes));
    }

    let mut logits = vec![0.0; trie.vocab_size()];
    let rec = Uppercase::new().append('N' as u8).append('E' as u8);
    for _ in 0..1000 {
        compute_bias(&trie, &rec, &mut logits);
    }

    println!("res: {}", logits.iter().filter(|x| **x > -50.0).count());
}
