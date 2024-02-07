use std::sync::Arc;

use crate::{config::ModelMeta, LoaderArgs, Repo, RllmEngine};
use anyhow::{bail, Result};

use llama_cpp_low as cpp;

use super::{
    blocks::CppBlockSpaceManager,
    tmodel::{CppLoaderArgs, TModel},
};

pub(super) fn load_rllm_engine(
    args: LoaderArgs,
    mut model_args: CppLoaderArgs,
) -> Result<RllmEngine<TModel>> {
    let model = do_load(&args, &mut model_args)?;
    let rllm_config = RllmEngine::<TModel>::build_config(&args, &mut model_args)?;

    let mut cparams = cpp::ContextParams::default();
    cparams.n_batch = rllm_config.scheduler.max_num_batched_tokens as u32;
    cparams.n_ctx = 10000; // TODO
    model.setup_context(cparams);

    let rllm_config = Arc::new(rllm_config);
    let tmodel = TModel::new(rllm_config.clone(), model);
    let block_mgr = CppBlockSpaceManager {};
    RllmEngine::build(args, tmodel, block_mgr, rllm_config)
}

fn do_load(args: &LoaderArgs, model_args: &mut CppLoaderArgs) -> Result<cpp::Model> {
    if model_args.cached_model.is_none() {
        let repo = Repo::from(args)?;
        log::info!("loading the model from {}", repo);

        let gguf = match args.file.as_ref() {
            Some(gguf) => gguf,
            None => {
                bail!("--gguf file.gguf or --model user/model::file.gguf is required for loading the model")
            }
        };

        let file = repo.get(gguf)?;

        let mut mparams = cpp::ModelParams::default();
        // TODO: make this configurable
        mparams.set_split_mode(cpp::SplitMode::None);
        mparams.n_gpu_layers = model_args.n_gpu_layers.unwrap_or(0) as i32;
        log::info!("{} layer(s) offloaded to GPU", mparams.n_gpu_layers);
        // don't GPU offload on Intel macs - it just fails there
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        {
            if mparams.n_gpu_layers > 0 {
                log::warn!("disabling GPU (Intel macOS)");
                mparams.n_gpu_layers = 0;
            }
        }

        let m = cpp::Model::from_file(file.to_str().unwrap(), mparams)?;
        model_args.cached_model = Some(m);
    }

    let model = model_args.cached_model.as_ref().unwrap().clone();
    Ok(model)
}

pub(super) fn load_model_config(
    args: &LoaderArgs,
    model_args: &mut CppLoaderArgs,
) -> Result<ModelMeta> {
    let model = do_load(args, model_args)?;

    let info = model.model_info();
    let vocab_size = info.n_vocab.try_into().unwrap();
    let max_sequence_length = info.n_ctx_train.try_into().unwrap();

    let mut meta = ModelMeta {
        id: args.model_id.clone(),
        max_sequence_length,
        vocab_size,
        tok_vocab_size: vocab_size,
    };

    // hidden_size: info.n_embd.try_into().unwrap(),
    // rope_theta: info.rope,
    // rotary_dim: max_sequence_length,

    let tok = aicirt::bintokens::find_tokenizer(&args.tokenizer)?;
    meta.tok_vocab_size = tok.tokrx_info().vocab_size as usize;

    Ok(meta)
}
