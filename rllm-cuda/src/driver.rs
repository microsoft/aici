use clap::Parser;
use rllm::util::parse_with_settings;

/// Serve LLMs with AICI over HTTP with tch (torch) backend.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct DriverArgs {
    #[clap(flatten)]
    pub args: rllm::server::RllmCliArgs,
}

#[actix_web::main]
async fn main() -> () {
    let args = parse_with_settings::<DriverArgs>();
    rllm::server::server_main(args.args).await;
}
