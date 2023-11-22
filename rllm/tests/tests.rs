use std::{collections::HashMap, sync::Mutex};

use actix_web::{http::header::ContentType, test, web::Data, App};
use candle_core::{DType, Device};
use candle_vllm::{
    get_model_loader,
    openai::{
        self, openai_server::chat_completions, requests::Messages, responses::APIError,
        OpenAIServerData,
    },
    ModelSelected,
};

#[actix_web::test]
async fn test_llama() -> Result<(), APIError> {
    let (loader, model_id) = get_model_loader(ModelSelected::Llama {
        no_kv_cache: false,
        repeat_last_n: 64,
        use_flash_attn: false,
    });
    let paths = loader.download_model(
        model_id,
        None,
        Some(std::env::var("TESTS_HF_TOKEN").unwrap()),
    )?;
    let model = loader.load_model(paths, DType::F16, Device::Cpu)?;

    let server_data = OpenAIServerData {
        pipeline_config: model.1,
        model: Mutex::new(model.0),
        device: Device::Cpu,
    };

    let app = test::init_service(
        App::new()
            .service(chat_completions)
            .app_data(Data::new(server_data)),
    )
    .await;

    let mut system = HashMap::new();
    system.insert("role".to_string(), "system".to_string());
    system.insert(
        "content".to_string(),
        "You are a talented author who specializes in writing poems.".to_string(),
    );

    let mut user = HashMap::new();
    user.insert("role".to_string(), "user".to_string());
    user.insert(
        "content".to_string(),
        "Please write me a poem about why Rust is a great programming language:".to_string(),
    );

    let req = test::TestRequest::with_uri("/v1/chat/completions")
        .insert_header(ContentType::json())
        .set_json(openai::requests::ChatCompletionRequest {
            model: "llama".to_string(),
            messages: Messages::Map(vec![system, user]),
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: None,
            stop: None,
            stream: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            top_k: None,
            best_of: None,
            use_beam_search: None,
            skip_special_tokens: None,
            ignore_eos: None,
            stop_token_ids: None,
        })
        .to_request();

    let resp = test::call_service(&app, req).await;
    println!("{:?}", resp.status());
    println!("{:?}", resp.into_body());
    Ok(())
}
