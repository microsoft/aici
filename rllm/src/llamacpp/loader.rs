use std::sync::Arc;

use crate::{
    config::{ModelConfig, ModelType},
    paged::CacheSize,
    LoaderArgs, Repo, RllmEngine, TModel,
};
use anyhow::{bail, Result};

use llama_cpp_low as cpp;

pub fn load_rllm_engine(mut args: LoaderArgs) -> Result<RllmEngine> {
    let model = do_load(&mut args)?;
    let rllm_config = RllmEngine::build_config(&mut args)?;
    let rllm_config = Arc::new(rllm_config);
    let tmodel = TModel::new(rllm_config.clone(), model);
    let cache_size = CacheSize { gpu: 0, cpu: 0 };
    RllmEngine::build(args, tmodel, rllm_config, cache_size)
}

fn do_load(args: &mut LoaderArgs) -> Result<cpp::Model> {
    if args.cached_model.is_none() {
        let repo = Repo::from(args)?;
        log::info!("loading the model from {}", repo);

        let gguf = match args.gguf.as_ref() {
            Some(gguf) => gguf,
            None => {
                bail!("--gguf file.gguf or --model user/model::file.gguf is required for loading the model")
            }
        };

        let file = repo.get(gguf)?;

        let mparams = cpp::ModelParams::default();
        let cparams = cpp::ContextParams::default();

        let m = cpp::Model::from_file(file.to_str().unwrap(), mparams, cparams)?;
        args.cached_model = Some(m);
    }

    let model = args.cached_model.as_ref().unwrap().clone();
    Ok(model)
}

pub fn load_model_config(args: &mut LoaderArgs) -> Result<ModelConfig> {
    let model = do_load(args)?;

    let common = args.common_config();
    let info = model.model_info();
    let vocab_size = info.n_vocab.try_into().unwrap();
    let max_sequence_length = info.n_ctx_train.try_into().unwrap();

    let mut v = ModelConfig {
        model_type: ModelType::LlamaCpp,
        meta: common.meta,
        hidden_size: info.n_embd.try_into().unwrap(),
        intermediate_size: 0,
        vocab_size,
        tok_vocab_size: vocab_size,
        num_hidden_layers: 0,
        num_attention_heads: 0,
        num_key_value_heads: 0,
        layer_norm_eps: 1e-5,
        rope_theta: info.rope,
        max_sequence_length,
        head_dim: 0,
        rotary_dim: max_sequence_length,
        dtype: common.dtype.unwrap_or(crate::DType::Float),
        device: common.device,
    };

    let tok = aicirt::bintokens::find_tokenizer(&args.tokenizer)?;
    v.tok_vocab_size = tok.tokrx_info().vocab_size as usize;

    Ok(v)
}
