use crate::{get_unix_time, InferenceResult};

use crate::openai::requests::CompletionRequest;
use crate::openai::responses::{
    APIError, ChatCompletionUsageResponse, CompletionResponse, StreamingCompletionChoice,
    StreamingCompletionResponse,
};
use crate::OpenAIServerData;

use actix_web::web::Bytes;
use actix_web::{post, web, Either, HttpResponse};
use aicirt::api::InstantiateReq;
use rllm::config::SamplingParams;
use rllm::seq::Token;
use rllm::AddRequest;
use serde_json::{json, Value};
use tokio::sync::mpsc::Receiver;
use uuid::Uuid;

fn check_length(
    request: &web::Json<CompletionRequest>,
    data: &OpenAIServerData,
) -> Result<Vec<Token>, APIError> {
    let token_ids = data
        .tokenizer
        .encode(request.prompt.clone(), true)
        .map_err(APIError::from)?
        .get_ids()
        .to_vec();

    let max_tokens = if let Some(max_toks) = request.max_tokens {
        max_toks
    } else {
        data.model_config.max_sequence_length - token_ids.len()
    };

    if token_ids.len() + max_tokens > data.model_config.max_sequence_length {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). Please reduce the length of the \
            messages or completion.",
            data.model_config.max_sequence_length,
            max_tokens + token_ids.len(),
            token_ids.len(),
            max_tokens
        )))
    } else {
        Ok(token_ids)
    }
}

macro_rules! set_fields_if_some {
    ($request:expr, $sampling_params:expr, $($field:ident),*) => {
        $(
            if let Some(v) = $request.$field {
                $sampling_params.$field = v;
            }
        )*
    };
}

macro_rules! bail_if_error {
    ($e:expr) => {
        if let Err(e) = $e {
            return Either::Left(Err(APIError::from(e)));
        }
    };
}

#[post("/v1/completions")]
async fn completions(
    data: web::Data<OpenAIServerData>,
    request: web::Json<CompletionRequest>,
) -> Either<Result<web::Json<CompletionResponse>, APIError>, HttpResponse> {
    if request.logit_bias.as_ref().is_some()
        && request.logit_bias.as_ref().is_some_and(|x| !x.is_empty())
    {
        return Either::Left(Err(APIError::new_str(
            "`logit_bias` is not currently supported.",
        )));
    }

    let token_ids = check_length(&request, &data);
    bail_if_error!(token_ids);

    let token_ids = token_ids.unwrap();

    let request_id = format!("cmpl-{}", Uuid::new_v4());

    let mut sampling_params = SamplingParams::default();

    set_fields_if_some!(
        request,
        sampling_params,
        n,
        best_of,
        presence_penalty,
        frequency_penalty,
        temperature,
        top_p,
        top_k,
        use_beam_search,
        ignore_eos,
        max_tokens
    );

    if request.best_of.is_none() {
        sampling_params.best_of = sampling_params.n;
    }

    if let Some(mod_id) = &request.aici_module {
        sampling_params.aici_module = Some(mod_id.clone());
        sampling_params.aici_arg = match &request.aici_arg {
            None => "".to_string(),
            Some(Value::String(s)) => s.clone(),
            Some(v) => serde_json::to_string(v).unwrap(),
        };
    }

    if let Some(stop) = request.stop.as_ref() {
        sampling_params.stop = stop.clone();
    }

    bail_if_error!(sampling_params.verify_args());

    if let Some(mod_id) = sampling_params.aici_module.as_ref() {
        let inst = data
            .side_cmd_ch
            .instantiate(InstantiateReq {
                req_id: request_id.clone(),
                prompt: json!(token_ids),
                module_id: mod_id.clone(),
                module_arg: json!(sampling_params.aici_arg),
            })
            .await;
        bail_if_error!(inst);
    }

    let rx = data.worker.lock().unwrap().add_request(AddRequest {
        request_id: request_id.clone(),
        prompt: token_ids,
        sampling_params,
        expected: None,
    });

    bail_if_error!(rx);
    let rx = rx.unwrap();

    let _stream = request.stream.is_some_and(|x| x);

    return Either::Right(
        HttpResponse::Ok()
            .append_header(("content-type", "text/event-stream"))
            //.no_chunking(asdf)
            .streaming(Client {
                rx,
                model: data.model_config.meta.id.clone(),
            }),
    );

    // Either::Left(Ok(web::Json(CompletionResponse {
    //     id: request_id,
    //     choices: result.0.unwrap(),
    //     created,
    //     model: request.model.clone(),
    //     object: "completion",
    //     usage: result.1,
    // })))
}

struct Client {
    rx: Receiver<InferenceResult>,
    model: String,
}

impl futures::Stream for Client {
    type Item = Result<Bytes, APIError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx).map(|x| match x {
            Some(Ok(so)) => {
                let u = &so.usage;
                let r = StreamingCompletionResponse {
                    object: "text_completion",
                    id: so.request_id,
                    model: self.model.clone(),
                    created: get_unix_time(),
                    usage: ChatCompletionUsageResponse {
                        completion_tokens: u.gen_tokens,
                        prompt_tokens: u.prompt_tokens,
                        total_tokens: u.total_tokens(),
                        fuel_tokens: u.fuel_tokens(),
                    },
                    choices: so
                        .seq_outputs
                        .iter()
                        .map(|choice| StreamingCompletionChoice {
                            text: choice.new_text.clone(),
                            index: choice.index,
                            finish_reason: choice.finish_reason.map(|r| r.short_name()),
                            logs: choice
                                .aici_logs
                                .iter()
                                .map(|e| e.logs.clone())
                                .collect::<Vec<_>>()
                                .join(""),
                            error: choice
                                .aici_logs
                                .iter()
                                .map(|e| e.error.clone())
                                .collect::<Vec<_>>()
                                .join(""),
                            storage: choice
                                .aici_logs
                                .iter()
                                .flat_map(|e| e.storage.clone())
                                .collect::<Vec<_>>(),
                        })
                        .collect(),
                };
                let res = serde_json::to_string(&r).unwrap();
                let mut res = format!("data: {}\n\n", res);
                if so.is_final {
                    res.push_str("data: [DONE]\n\n");
                }
                Some(Ok(Bytes::from(res)))
            }
            Some(Err(e)) => Some(Err(APIError::from(e))),
            None => None,
        })
    }
}
