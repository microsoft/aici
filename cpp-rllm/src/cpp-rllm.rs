use clap::Parser;
use rllm::util::parse_with_settings;

/// Serve LLMs with AICI over HTTP with llama.cpp backend.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CppArgs {
    #[clap(flatten)]
    pub args: rllm::server::RllmCliArgs,

    /// Name of .gguf file inside of the model folder/repo.
    #[arg(long, help_heading = "Model")]
    pub gguf: Option<String>,
}

#[actix_web::main]
async fn main() -> () {
    let mut args = parse_with_settings::<CppArgs>();
    args.args.gguf = args.gguf;
    rllm::server::server_main(args.args).await;
}
