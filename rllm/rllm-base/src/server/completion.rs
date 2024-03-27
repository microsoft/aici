use crate::seq::{FinishReason, RequestOutput, SeqOutput};
use crate::server::{auth_info, APIError, AiciServerData, InferenceResult};
use crate::{config::SamplingParams, seq::Token, AddRequest};
use actix_web::{post, web, web::Bytes, HttpResponse};
use aicirt::{api::InstantiateReq, get_unix_time};
use serde_json::{json, Value};
use tokio::sync::mpsc::Receiver;
use uuid::Uuid;

use super::api::{InitialRunResponse, RunForkResponse, RunRequest, RunResponse, RunUsageResponse};

const NONE_CONTROLLER: &str = "none";

fn check_length(
    request: &web::Json<RunRequest>,
    data: &AiciServerData,
) -> Result<(usize, Vec<Token>), APIError> {
    let prompt = if request.controller == NONE_CONTROLLER {
        request.controller_arg.as_str().unwrap_or("")
    } else {
        ""
    };
    let token_ids = data
        .tokenizer
        .encode(prompt, true)
        .map_err(APIError::from)?
        .get_ids()
        .to_vec();

    let max_tokens = if let Some(max_toks) = request.max_tokens {
        max_toks
    } else {
        data.model_meta.max_sequence_length - token_ids.len()
    };

    if token_ids.len() + max_tokens > data.model_meta.max_sequence_length {
        Err(APIError::new(format!(
            "This model's maximum context length is {} tokens. \
            However, you requested {} tokens ({} in the messages, \
            {} in the completion). Please reduce the length of the \
            messages or completion.",
            data.model_meta.max_sequence_length,
            max_tokens + token_ids.len(),
            token_ids.len(),
            max_tokens
        )))
    } else {
        Ok((max_tokens, token_ids))
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
            return Err(APIError::from(e));
        }
    };
}

#[post("/v1/run")]
async fn run_controller(
    req: actix_web::HttpRequest,
    data: web::Data<AiciServerData>,
    request: web::Json<RunRequest>,
) -> Result<HttpResponse, APIError> {
    let token_ids = check_length(&request, &data);
    bail_if_error!(token_ids);

    let (max_tokens, mut token_ids) = token_ids.unwrap();

    let request_id = format!("run-{}", Uuid::new_v4());

    let mut sampling_params = SamplingParams::default();
    sampling_params.max_tokens = max_tokens;
    sampling_params.ignore_eos = true;

    set_fields_if_some!(request, sampling_params, temperature, top_p, top_k);

    if request.controller != NONE_CONTROLLER {
        sampling_params.controller = Some(request.controller.clone());
        sampling_params.controller_arg = match &request.controller_arg {
            Value::String(s) => s.clone(),
            v => serde_json::to_string(v).unwrap(),
        };
    }

    bail_if_error!(sampling_params.verify_args());

    let init_result = if let Some(mod_id) = sampling_params.controller.as_ref() {
        let inst = data
            .side_cmd_ch
            .instantiate(
                InstantiateReq {
                    req_id: request_id.clone(),
                    prompt: json!(token_ids),
                    module_id: mod_id.clone(),
                    module_arg: json!(sampling_params.controller_arg),
                },
                auth_info(&req),
            )
            .await;
        bail_if_error!(inst);
        Some(inst.unwrap())
    } else {
        None
    };

    let rx = match init_result {
        Some(r) if r.error.len() > 0 => {
            let outp = RequestOutput {
                request_id: request_id.clone(),
                usage: Default::default(),
                seq_outputs: vec![SeqOutput {
                    seq_id: 0,
                    index: 0,
                    new_output_tokens: vec![],
                    new_text: String::new(),
                    output_tokens: vec![],
                    finish_reason: Some(FinishReason::Failed),
                    aici_logs: vec![r],
                }],
                is_final: true,
            };
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tx.send(Ok(outp)).await.unwrap();
            rx
        }
        _ => {
            let rx = data.worker.lock().unwrap().add_request(AddRequest {
                request_id: request_id.clone(),
                prompt: token_ids,
                sampling_params,
                expected: None,
                init_result,
            });

            bail_if_error!(rx);
            rx.unwrap()
        }
    };

    return Ok(HttpResponse::Ok()
        .append_header(("content-type", "text/event-stream"))
        .streaming(Client {
            rx,
            initial: Some(InitialRunResponse {
                id: request_id,
                object: "initial-run",
                created: get_unix_time(),
                model: data.model_meta.id.clone(),
            }),
        }));
}

struct Client {
    initial: Option<InitialRunResponse>,
    rx: Receiver<InferenceResult>,
}

impl futures::Stream for Client {
    type Item = Result<Bytes, APIError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let init = std::mem::take(&mut self.initial);
        match init {
            Some(init) => {
                return std::task::Poll::Ready(Some(Ok(Bytes::from(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&init).unwrap()
                )))));
            }
            None => {}
        }

        self.rx.poll_recv(cx).map(|x| match x {
            Some(Ok(so)) => {
                let u = &so.usage;
                let r = RunResponse {
                    object: "run",
                    usage: RunUsageResponse {
                        sampled_tokens: u.gen_tokens,
                        ff_tokens: u.prompt_tokens,
                        cost: u.fuel_tokens(),
                    },
                    forks: so
                        .seq_outputs
                        .iter()
                        .map(|choice| RunForkResponse {
                            text: choice.new_text.clone(),
                            index: choice.index,
                            finish_reason: choice.finish_reason.map(|r| r.short_name()),
                            micros: choice.aici_logs.iter().map(|e| e.micros).sum(),
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
