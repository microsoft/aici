use crate::{InferenceResult, get_unix_time};

use crate::openai::requests::CompletionRequest;
use crate::openai::responses::{
    APIError, CompletionResponse, StreamingCompletionChoice, StreamingCompletionResponse,
};
use crate::OpenAIServerData;

use actix_web::web::Bytes;
use actix_web::{post, web, Either, HttpResponse};
use rllm::config::SamplingParams;
use rllm::seq::Token;
use rllm::AddRequest;
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
    if token_ids.is_err() {
        return Either::Left(Err(token_ids.err().unwrap()));
    }
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

    if let Some(stop) = request.stop.as_ref() {
        sampling_params.stop = stop.clone();
    }

    if let Err(e) = sampling_params.verify_args() {
        return Either::Left(Err(APIError::from(e)));
    }

    let rx = data.worker.lock().unwrap().add_request(AddRequest {
        request_id: request_id.clone(),
        prompt: token_ids,
        sampling_params,
    });

    if let Err(e) = rx {
        return Either::Left(Err(APIError::from(e)));
    }
    let rx = rx.unwrap();

    let _stream = request.stream.is_some_and(|x| x);

    return Either::Right(
        HttpResponse::Ok()
            .append_header(("content-type", "text/event-stream"))
            //.no_chunking(asdf)
            .streaming(Client(rx)),
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

struct Client(Receiver<InferenceResult>);

impl futures::Stream for Client {
    type Item = Result<Bytes, APIError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.0.poll_recv(cx).map(|x| match x {
            Some(Ok(so)) => {
                let r = StreamingCompletionResponse {
                    object: "text_completion",
                    id: so.request_id,
                    model: "current".to_string(),
                    created: get_unix_time(),
                    choices: so
                        .seq_outputs
                        .iter()
                        .map(|choice| StreamingCompletionChoice {
                            text: choice.new_text.clone(),
                            index: choice.index,
                            finish_reason: choice.finish_reason.map(|r| r.short_name()),
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
