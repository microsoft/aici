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
    let _ = args;

    // let dtype = match args.args.dtype.as_str() {
    //     "bf16" => Some(DType::BFloat16),
    //     "f16" => Some(DType::Half),
    //     "f32" => Some(DType::Float),
    //     "" => None,
    //     _ => panic!("invalid dtype; try one of bf16, f16, f32"),
    // };

    // rllm::server::server_main::<TModel>(args.args).await;
}
