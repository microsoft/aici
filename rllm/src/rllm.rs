use std::collections::HashMap;

use aicirt::setup_log;
use anyhow::Result;
use clap::Parser;

use rand::Rng;
use rllm::{config::SamplingParams, playground_1, AddRequest, LoaderArgs, RllmEngine};

const DEFAULT_PROMPT: &str = "over millions of years,";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 10)]
    sample_len: usize,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// File with prompt.
    #[arg(long)]
    prompt_file: Option<String>,

    #[arg(long)]
    model_id: String,

    #[arg(long)]
    revision: Option<String>,

    /// Tokenizer to use; try --tokenizer list to see options
    #[arg(short, long, default_value = "llama")]
    tokenizer: String,

    #[arg(long, default_value_t = 0)]
    alt: usize,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,

    #[arg(long, default_value_t = false)]
    nv_profile: bool,
}

fn main() -> Result<()> {
    setup_log();

    let args = Args::parse();

    if args.alt == 3 {
        playground_1();
        return Ok(());
    }

    let mut engine = RllmEngine::load(LoaderArgs {
        model_id: args.model_id,
        revision: args.revision,
        local_weights: args.local_weights,
        alt: args.alt,
        tokenizer: args.tokenizer,
        ..Default::default()
    })?;

    let prompt = match &args.prompt {
        Some(p) => p.clone(),
        None => match &args.prompt_file {
            Some(f) => std::fs::read_to_string(f)?,
            None => DEFAULT_PROMPT.to_string(),
        },
    };

    println!("{prompt}");

    let mut p = SamplingParams::default();
    p.temperature = args.temperature.map_or(p.temperature, |v| v as f32);
    p.top_p = args.top_p.map_or(p.top_p, |v| v as f32);
    p.max_tokens = args.sample_len;

    let start_gen = std::time::Instant::now();

    if args.alt == 7 {
        // "the color of nature, the color of the earth",
        engine.add_request("R1".to_string(), "Color green is", p.clone())?;
        // "Alfred Tarski in 1936."
        engine.add_request(
            "R2".to_string(),
            "Tarski's fixed-point theorem was proven by",
            p.clone(),
        )?;
        // "the existence of a fixed point in a certain relation",
        engine.add_request(
            "R3".to_string(),
            "Tarski's fixed-point theorem is about",
            p.clone(),
        )?;

        for _ in 0..1 {
            let res = engine.step().unwrap();
            for sgo in &res {
                assert!(sgo.seq_outputs.len() == 1);
                let so = &sgo.seq_outputs[0];
                let t = engine.seq_output_text(so)?;
                let rid = &sgo.request_id;
                println!("{rid} {t}");
            }
        }
    } else if args.alt == 8 {
        let s = String::from_utf8(std::fs::read("words.txt").unwrap()).unwrap();
        let words: Vec<_> = s.split_whitespace().collect();
        let mut rng = rand::thread_rng();

        let mut idx = 0;
        let l = args.sample_len;
        let mut prompts = HashMap::new();
        loop {
            while engine.num_pending_requests() < 30 {
                let start = rng.gen_range(0..words.len() - 50);
                let len = rng.gen_range(4..10);
                let prompt = words[start..start + len].join(" ");
                let id = format!("R{}", idx);
                prompts.insert(id.clone(), prompt.clone());
                engine.add_request(id, &prompt, p.clone())?;
                idx += 1;
                eprint!("*");
            }
            let res = engine.step().unwrap();
            for sgo in &res {
                assert!(sgo.seq_outputs.len() == 1);
                if sgo.is_ambiguous {
                    engine.abort_request(&sgo.request_id);
                } else {
                    let so = &sgo.seq_outputs[0];
                    if so.output_tokens.len() >= l {
                        let t = engine.seq_output_text(so)?;
                        println!("\nFOUND {:?} {:?}", prompts[&sgo.request_id], t);
                        engine.abort_request(&sgo.request_id);
                    }
                }
            }
        }
    } else if args.alt == 9 {
        let tokens = engine.tokenize(&prompt, true)?;
        let mut len = 10;
        while len < tokens.len() {
            let tokens = &tokens[..len];
            engine.queue_request(AddRequest {
                request_id: format!("R{len}"),
                prompt: tokens.to_vec(),
                sampling_params: p.clone(),
            })?;
            while engine.num_pending_requests() > 0 {
                let _ = engine.step()?;
            }
            len = len / 10 * 11;
        }
    } else {
        if args.nv_profile {
            log::info!("pre-heating...");
            let mut pre_heat = p.clone();
            pre_heat.max_tokens = 1;
            let _ = engine.generate(&prompt, pre_heat)?;
            log::info!("pre-heating done");
        }

        engine.nv_profile = args.nv_profile;

        let gen = engine.generate(&prompt, p)?;
        let dt = start_gen.elapsed();
        println!("\n{gen}\n");
        log::info!("time: {dt:?}");
    }

    Ok(())
}
