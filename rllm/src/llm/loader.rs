use crate::{
    config::{ModelConfig, ModelType, RllmConfig},
    llm::{
        llama, phi,
        seqid::SeqIdGen,
        tmodel::TModel,
        util::{gpu_memory_size, gpu_peak_allocated_bytes, log_mem_stats, reset_mem_stats},
    },
    paged::{BatchInfoBuilder, CacheEngine, CacheSize},
    DType, HashSet, LoaderArgs, Repo, RllmEngine, RllmModelConfig,
};
use anyhow::{bail, Result};
use safetensors::Dtype;
use std::{path::PathBuf, rc::Rc, sync::Arc};
use tch::{nn::VarStore, Device, Kind, Tensor};

use super::tmodel::TModelInner;

fn kind_from_dt(dtype: Dtype) -> Kind {
    match dtype {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        dtype => panic!("unsupported dtype {dtype:?}"),
    }
}

fn read_tensor(s: &safetensors::SafeTensors, name: &str) -> Result<Tensor> {
    let view = s.tensor(name)?;
    let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
    let kind: DType = kind_from_dt(view.dtype());
    // Using from_blob here instead of from_data_size avoids some unnecessary copy.
    let tensor = unsafe { Tensor::from_blob(view.data().as_ptr(), &size, &[], kind, Device::Cpu) };
    Ok(tensor)
}

fn load_model(rllm_config: &RllmConfig, filenames: Vec<PathBuf>) -> Result<Box<dyn TModelInner>> {
    let mut vs = VarStore::new(rllm_config.device.clone());

    let rc_cfg = Rc::new(rllm_config.model.clone());
    let mut model: Box<dyn TModelInner> = match rllm_config.model.model_type {
        ModelType::Llama => Box::new(llama::Llama::load(vs.root(), &rc_cfg).unwrap()),
        ModelType::Phi => Box::new(phi::MixFormerSequentialForCausalLM::new(&rc_cfg, vs.root())),
        ModelType::LlamaCpp => {
            panic!("LlamaCpp model type is not supported by the Rust loader")
        }
    };

    vs.set_kind(rllm_config.dtype);

    let mut vars = vs.variables();

    let bar = indicatif::ProgressBar::new(vars.len() as u64);
    bar.set_style(
        indicatif::ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:60.cyan/blue} {pos:>4}/{len:4} [{eta_precise}] {msg}",
        )
        .unwrap(),
    );

    for f in &filenames {
        let fp = std::fs::File::open(f)?;
        let content = unsafe { memmap2::MmapOptions::new().map(&fp)? };
        let safetensors = safetensors::SafeTensors::deserialize(&content)?;

        for vname in safetensors.names() {
            let target_name = vname.to_string();
            if !vars.contains_key(&target_name) {
                if vname.ends_with(".inv_freq") {
                    // OK
                } else {
                    log::warn!("variable {} not found in the model", target_name);
                }
                continue;
            }

            // Using from_blob here instead of from_data_size avoids some unnecessary copy.
            let src_tensor = read_tensor(&safetensors, vname)?;
            let mut var = vars.remove(&target_name).unwrap();
            assert!(var.size() == src_tensor.size());
            // println!("copying to {var:?} from {src_tensor:?}");
            var.f_copy_(&src_tensor)?;

            bar.inc(1);
            if bar.is_hidden() {
                eprint!(".");
            }
        }
    }

    if vars.len() > 0 {
        bail!("{} variables not found in the model: {vars:?}", vars.len());
    }

    if bar.is_hidden() {
        eprintln!(" done");
    }
    bar.finish();

    log::info!("model loaded");

    model.finalize();

    Ok(model)
}

fn model_filenames(repo: &Repo) -> Result<Vec<PathBuf>> {
    let idx = repo.read("model.safetensors.index.json");

    let filenames = if let Ok(idx) = idx {
        let st_index: serde_json::Value = serde_json::from_slice(&idx)?;
        let entries = st_index["weight_map"]
            .as_object()
            .unwrap()
            .values()
            .map(|v| v.as_str().unwrap().to_owned());

        let h = HashSet::<String>::from_iter(entries);
        let mut filenames = h.into_iter().collect::<Vec<_>>();
        filenames.sort();
        filenames
    } else {
        if repo.is_local() && repo.get("model.safetensors-rust").is_ok() {
            vec!["model.safetensors-rust".to_string()]
        } else {
            vec!["model.safetensors".to_string()]
        }
    };

    let filenames = filenames
        .iter()
        .map(|f| repo.get(f))
        .collect::<Result<Vec<_>>>()?;

    Ok(filenames)
}

pub fn load_rllm_engine(mut args: LoaderArgs) -> Result<RllmEngine> {
    let _no_grad = tch::no_grad_guard();

    let device = args.device;
    let repo = Repo::from(&args)?;

    let rllm_config = RllmEngine::build_config(&mut args)?;

    let filenames = model_filenames(&repo)?;
    log::info!("building the model");

    let _ = Tensor::zeros(&[1], (rllm_config.model.dtype, device));
    reset_mem_stats(device);
    log_mem_stats("initial", device);

    let model = load_model(&rllm_config, filenames)?;

    log_mem_stats("model fully loaded", device);

    let rllm_config = Arc::new(rllm_config);
    let cache_size = profile_model(rllm_config.clone(), &model);
    let cache_engine = CacheEngine::new(rllm_config.clone(), &cache_size);

    let tmodel = TModel::new(rllm_config.clone(), cache_engine, model);

    RllmEngine::build(args, tmodel, rllm_config, cache_size, SeqIdGen::new())
}

fn profile_model(config: Arc<RllmConfig>, model: &Box<dyn TModelInner>) -> CacheSize {
    let device = config.device.clone();
    let gpu_mem = gpu_memory_size(device);

    let gpu_cache_size = if gpu_mem > 0 {
        let mut info = BatchInfoBuilder::new(config.clone()).profile_run();
        reset_mem_stats(device);
        log_mem_stats("before model profile", device);
        let _logits = model.forward(&mut info);
        log_mem_stats("after model profile", device);

        let frac = config.cache.gpu_memory_utilization;
        let peak = gpu_peak_allocated_bytes(device) as isize;
        let left = (gpu_mem as f64 * frac) as isize - peak;
        if left < 0 {
            panic!("not enough GPU memory for the cache: {gpu_mem} * {frac} < {peak}");
        }
        left as usize
    } else {
        512 << 20 // 512MiB
    };

    let max_cpu = 2 << 30; // 2GiB
    let cpu_cache_size = std::cmp::min(max_cpu, gpu_cache_size);

    let elt_size = CacheEngine::get_cache_block_size(&config);

    let r = CacheSize {
        cpu: cpu_cache_size / elt_size,
        gpu: gpu_cache_size / elt_size,
    };

    let token_kv_size = elt_size / config.cache.block_size;

    const G: f64 = 1024.0 * 1024.0 * 1024.0;
    log::info!(
        "caches: gpu:{:.3}GiB cpu:{:.3}GiB; blocks: {}/{}; tokens: {}/{}; {}KiB/token",
        gpu_cache_size as f64 / G,
        cpu_cache_size as f64 / G,
        r.gpu,
        r.cpu,
        r.gpu * config.cache.block_size,
        r.cpu * config.cache.block_size,
        token_kv_size / 1024,
    );

    r
}

pub fn load_model_config(args: &LoaderArgs) -> Result<ModelConfig> {
    let repo = Repo::from(args)?;
    log::info!("loading the model from {}", repo);

    let bytes = repo.read("config.json")?;
    let mut err = String::new();

    let cfg = load_one_config::<llama::LlamaConfig>(&mut err, args, "llama", &bytes)
        .or_else(|| load_one_config::<phi::PhiConfig>(&mut err, args, "phi", &bytes));

    match cfg {
        Some(mut v) => {
            let tok = aicirt::bintokens::find_tokenizer(&args.tokenizer)?;
            v.tok_vocab_size = tok.tokrx_info().vocab_size as usize;
            Ok(v)
        }
        None => bail!("failed to load model config:\n{}", err),
    }
}

fn load_one_config<T>(
    err: &mut String,
    args: &LoaderArgs,
    name: &str,
    bytes: &[u8],
) -> Option<ModelConfig>
where
    T: RllmModelConfig + serde::de::DeserializeOwned,
{
    let common = args.common_config();
    let json = serde_json::from_slice::<T>(bytes);
    if let Ok(json) = json {
        Some(json.into_config(common))
    } else {
        *err += &format!("{name}: {}\n", json.err().unwrap());
        None
    }
}
