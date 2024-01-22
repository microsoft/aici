#[actix_web::main]
async fn main() -> () {
    rllm::server::server_main().await;
}
