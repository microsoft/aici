use std::thread;

use super::requests::ChatCompletionRequest;
use super::requests::Messages;
use super::responses::{APIError, ChatCompletionResponse};
use super::sampling_params::{EarlyStoppingCondition, SamplingParams};
use super::streaming::new_streaming_conn;
use super::utils::get_created_time_secs;
use super::OpenAIServerData;
use actix_web::web::Bytes;
use actix_web::{post, web, Either, HttpResponse};
use tokenizers::Encoding;
use uuid::Uuid;

fn verify_model(data: &OpenAIServerData<'_>, model_name: &String) -> Result<(), APIError> {
    let current_name = {
        let model = data.model.lock().unwrap();
        model.name().to_string()
    };
    if &current_name != model_name {
        Err(APIError::new(format!(
            "Model name `{model_name}` is invalid."
        )))
    } else {
        Ok(())
    }
}

// Get prompt, roles
async fn get_gen_prompt(
    data: &OpenAIServerData<'_>,
    request: &web::Json<ChatCompletionRequest>,
) -> Result<String, APIError> {
    let mut model = data.model.lock().unwrap();
    let conversation = model.get_conversation();

    match &request.messages {
        Messages::Literal(msg) => {
            return Ok(msg.clone());
        }
        Messages::Map(messages) => {
            for message in messages {
                let role = message
                    .get("role")
                    .ok_or(APIError::new("Message key `role` not found.".to_string()))?;
                let content = message
                    .get("content")
                    .ok_or(APIError::new(
                        "Message key `content` not found.".to_string(),
                    ))?
                    .clone();

                if role == "system" {
                    conversation.set_system_message(content);
                } else if role == "user" {
                    conversation.append_message(conversation.get_roles().0.clone(), content)
                } else if role == "assistant" {
                    conversation.append_message(conversation.get_roles().1.clone(), content)
                } else {
                    return Err(APIError::new(format!("Unknown role: {role}")));
                }
            }
        }
    }

    conversation.append_none_message(conversation.get_roles().1.clone());

    Ok(conversation.get_prompt())
}

fn check_length(
    request: &web::Json<ChatCompletionRequest>,
    prompt: String,
    data: &OpenAIServerData<'_>,
) -> Result<Encoding, APIError> {
    let token_ids = {
        let model = data.model.lock().unwrap();
        model.tokenizer().tokenize(prompt)?
    };

    let max_tokens = if let Some(max_toks) = request.max_tokens {
        max_toks
    } else {
        data.pipeline_config.max_model_len - token_ids.len()
    };

    if token_ids.len() + max_tokens > data.pipeline_config.max_model_len {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). Please reduce the length of the \
            messages or completion.",
            data.pipeline_config.max_model_len,
            max_tokens + token_ids.len(),
            token_ids.len(),
            max_tokens
        )))
    } else {
        Ok(token_ids)
    }
}

#[post("/v1/chat/completions")]
async fn chat_completions(
    data: web::Data<OpenAIServerData<'static>>,
    request: web::Json<ChatCompletionRequest>,
) -> Either<Result<web::Json<ChatCompletionResponse>, APIError>, HttpResponse> {
    let model_name = &request.model;
    let res = verify_model(&data, model_name);
    if res.is_err() {
        return Either::Left(Err(res.err().unwrap()));
    }

    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return Either::Left(Err(APIError::new_str(
            "`logit_bias` is not currently supported.",
        )));
    }

    let prompt = get_gen_prompt(&data, &request).await;
    if prompt.is_err() {
        return Either::Left(Err(prompt.err().unwrap()));
    }
    let prompt = prompt.unwrap();

    let token_ids = check_length(&request, prompt, &data);
    if token_ids.is_err() {
        return Either::Left(Err(token_ids.err().unwrap()));
    }
    let token_ids = token_ids.unwrap();

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let sampling_params = SamplingParams::new(
        request.n.unwrap_or(1),
        request.best_of,
        request.presence_penalty.unwrap_or(0.0),
        request.frequency_penalty.unwrap_or(0.0),
        1.0,
        request.temperature.unwrap_or(0.7),
        request.top_p.unwrap_or(1.0),
        request.top_k.unwrap_or(-1),
        request.use_beam_search.unwrap_or(false),
        1.0,
        EarlyStoppingCondition::UnlikelyBetterCandidates,
        request.stop.clone(),
        request.stop_token_ids.clone().unwrap_or_default(),
        request.ignore_eos.unwrap_or(false),
        request.max_tokens.unwrap_or(16),
        None,
        None,
        request.skip_special_tokens.unwrap_or(true),
    );
    if sampling_params.is_err() {
        return Either::Left(Err(sampling_params.err().unwrap()));
    }
    let sampling_params = sampling_params.unwrap();

    let created = get_created_time_secs();

    if request.stream.is_some_and(|x| x) {
        let (sender, receiver) = new_streaming_conn();
        let _ = thread::spawn(move || {
            let mut model = data.model.lock().unwrap();
            let model_res = model.forward(
                &token_ids,
                sampling_params,
                data.device.clone(),
                Some(sender.clone()),
            );
            if model_res.is_err() {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();

                // Ignore sending errors
                let _ = runtime.block_on(sender.send(Ok(Bytes::from(
                    serde_json::to_vec(&model_res.err().unwrap()).unwrap(),
                ))));
            }
        });

        return Either::Right(
            HttpResponse::Ok()
                .append_header(("content-type", "text/event-stream"))
                //.no_chunking(asdf)
                .streaming(receiver),
        );
    }

    let result = {
        let mut model = data.model.lock().unwrap();
        let model_res = model.forward(&token_ids, sampling_params, data.device.clone(), None);
        if model_res.is_err() {
            return Either::Left(Err(model_res.err().unwrap()));
        }
        model_res.unwrap()
    };

    Either::Left(Ok(web::Json(ChatCompletionResponse {
        id: request_id,
        choices: result.0.unwrap(),
        created,
        model: request.model.clone(),
        object: "chat.completion",
        usage: result.1,
    })))
}
