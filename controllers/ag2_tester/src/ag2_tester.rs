use aici_ag2_ctrl::earley::bench::earley_test;
use aici_native::bintokens::{find_tokenizer, guess_tokenizer, ByteTokenizerEnv};

fn main() {
    let tokenizer = guess_tokenizer("orca").unwrap();
    let tok = find_tokenizer(&tokenizer).unwrap();
    let env = ByteTokenizerEnv::new(tok);
    earley_test(env.tok_trie)
}
