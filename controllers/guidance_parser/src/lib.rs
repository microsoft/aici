use std::sync::Mutex;

use aici_abi::{set_host, toktree::TokTrie, HostInterface, StorageCmd, StorageResp, TokenId};
use aici_native::{
    bintokens::{self, ByteTokenizer}, setup_log, variables::Variables
};
use anyhow::Result;

struct ParserHost {
    trie_bytes: Vec<u8>,
    tokenizer: ByteTokenizer,
    vars: Mutex<Variables>,
}

impl HostInterface for ParserHost {
    fn arg_bytes(&self) -> Vec<u8> {
        todo!()
    }

    fn trie_bytes(&self) -> Vec<u8> {
        self.trie_bytes.clone()
    }

    fn return_logit_bias(&self, _vob: &aici_abi::svob::SimpleVob) {
        todo!()
    }

    fn process_arg_bytes(&self) -> Vec<u8> {
        todo!()
    }

    fn return_process_result(&self, _res: &[u8]) {
        todo!()
    }

    fn storage_cmd(&self, cmd: StorageCmd) -> StorageResp {
        let mut vars = self.vars.lock().unwrap();
        vars.process_cmd(cmd)
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        let tokens = self
            .tokenizer
            .hf_tokenizer
            .encode(String::from_utf8_lossy(s), false);
        match tokens {
            Err(e) => panic!("tokenize error: {e}"),
            Ok(tokens) => Vec::from(tokens.get_ids()),
        }
    }

    fn self_seq_id(&self) -> aici_abi::SeqId {
        aici_abi::SeqId(42)
    }

    fn eos_token(&self) -> TokenId {
        self.tokenizer.eos_token
    }

    fn stop(&self) -> ! {
        panic!("AICI stop called")
    }
}

pub fn init(tokenizer_name: &str) -> Result<()> {
    setup_log();
    let tokenizer = bintokens::find_tokenizer(tokenizer_name)?;
    let tokens = tokenizer.token_bytes();
    let trie = TokTrie::from(&tokenizer.tokrx_info(), &tokens);
    trie.check_against(&tokens);

    set_host(Box::new(ParserHost {
        tokenizer,
        trie_bytes: trie.serialize(),
        vars: Mutex::new(Variables::default()),
    }));

    Ok(())
}
