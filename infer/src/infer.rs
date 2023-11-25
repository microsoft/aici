use anyhow::Result;
use clap::Parser;

use rllm::{LlamaInfer, LoaderArgs, LogitsProcessor};

const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 10)]
    sample_len: usize,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut infer = LlamaInfer::load(LoaderArgs {
        model_id: args.model_id,
        revision: args.revision,
        local_weights: args.local_weights,
        use_reference: true,
    })?;

    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());

    println!("{prompt}");

    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);

    let start_gen = std::time::Instant::now();
    let gen = infer.generate(prompt, args.sample_len, &mut logits_processor)?;
    let dt = start_gen.elapsed();
    println!("\n{gen}\ntime: {dt:?}\n");
    Ok(())
}
