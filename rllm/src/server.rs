use anyhow::Result;
use clap::Parser;

use rllm::{LoaderArgs, RllmEngine};

mod openai;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Huggingface model name
    #[arg(long)]
    model_id: Option<String>,

    /// Huggingface model revision
    #[arg(long)]
    revision: Option<String>,

    #[arg(long, default_value_t = false)]
    reference: bool,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
}

fn main() -> Result<()> {
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.init();

    let args = Args::parse();

    let mut engine = RllmEngine::load(LoaderArgs {
        model_id: args.model_id,
        revision: args.revision,
        local_weights: args.local_weights,
        use_reference: args.reference,
        alt: 0,
    })?;

    // let gen = engine.generate(prompt, p)?;

    Ok(())
}
