use crate::{config::ModelConfig, LoaderArgs, Repo, RllmEngine};
use anyhow::{bail, Result};

pub fn load_rllm_engine(args: LoaderArgs) -> Result<RllmEngine> {
    todo!()
}

pub fn load_model_config(args: &LoaderArgs) -> Result<ModelConfig> {
    let repo = Repo::from(args)?;
    log::info!("loading the model from {}", repo);

    let gguf = match args.gguf.as_ref() {
        Some(gguf) => gguf,
        None => {
            bail!("--gguf file.gguf or --model user/model::file.gguf is required for loading the model")
        }
    };

    let file = repo.get(gguf)?;

    todo!()

    // match cfg {
    //     Some(mut v) => {
    //         let tok = aicirt::bintokens::find_tokenizer(&args.tokenizer)?;
    //         v.tok_vocab_size = tok.tokrx_info().vocab_size as usize;
    //         Ok(v)
    //     }
    //     None => bail!("failed to load model config:\n{}", err),
    // }
}
