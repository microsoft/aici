use clap::Parser;
use rllm::{
    llamacpp::tmodel::{CppLoaderArgs, TModel},
    util::parse_with_settings,
};

/// Serve LLMs with AICI over HTTP with llama.cpp backend.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CppArgs {
    #[clap(flatten)]
    pub args: rllm::server::RllmCliArgs,

    /// Name of .gguf file inside of the model folder/repo.
    #[arg(long, help_heading = "Model")]
    pub gguf: Option<String>,

    /// How many model layers to offload to GPU (if available)
    #[arg(long, short = 'g', help_heading = "Model")]
    pub gpu_layers: Option<usize>,
}

#[actix_web::main]
async fn main() -> () {
    let mut args = parse_with_settings::<CppArgs>();
    args.args.file = args.gguf;
    let model_args = CppLoaderArgs::new(args.gpu_layers);
    rllm::server::server_main::<TModel>(args.args, model_args).await;
}
