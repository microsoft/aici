use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use actix_web::{middleware::Logger, web, App, HttpServer};
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{MkModuleReq, MkModuleResp},
    setup_log,
};
use anyhow::Result;
use base64::Engine;
use clap::Parser;

use rllm::{
    config::{ModelConfig, SamplingParams},
    iface::{AiciRtIface, AsyncCmdChannel},
    seq::RequestOutput,
    AddRequest, LoaderArgs, RllmEngine,
};

use openai::responses::APIError;
use tokio::sync::mpsc::{channel, error::TryRecvError, Receiver, Sender};

mod completion;
mod openai;

#[derive(Clone)]
pub struct OpenAIServerData {
    pub worker: Arc<Mutex<InferenceWorker>>,
    pub model_config: ModelConfig,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub tok_trie: Arc<TokTrie>,
    pub side_cmd_ch: AsyncCmdChannel,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Port to serve on (localhost:port)
    #[arg(long)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Huggingface model name
    #[arg(long)]
    model: String,

    /// Huggingface model revision; --model foo/bar@revision is also possible
    #[arg(long)]
    revision: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,

    /// Tokenizer to use; try --tokenizer list to see options
    #[arg(short, long, default_value = "llama")]
    tokenizer: String,

    /// Path to the aicirt binary.
    #[arg(long)]
    aicirt: String,

    /// Size of JSON comm buffer in megabytes
    #[arg(long, default_value = "32")]
    json_size: usize,

    /// Size of binary comm buffer in megabytes
    #[arg(long, default_value = "32")]
    bin_size: usize,

    /// How many milliseconds to spin-wait for a message over IPC and SHM.
    #[arg(long, default_value = "200")]
    busy_wait_time: u64,

    /// Shm/semaphore name prefix
    #[arg(long, default_value = "/aici0-")]
    shm_prefix: String,

    /// Enable nvprof profiling for given engine step
    #[arg(long, default_value_t = 0)]
    profile_step: usize,
}

#[actix_web::post("/v1/aici_modules")]
async fn upload_aici_module(
    data: web::Data<OpenAIServerData>,
    body: web::Bytes,
) -> Result<web::Json<MkModuleResp>, APIError> {
    body.len();
    let binary = base64::engine::general_purpose::STANDARD.encode(body);
    let r = data
        .side_cmd_ch
        .mk_module(MkModuleReq {
            binary,
            meta: serde_json::Value::Null,
        })
        .await
        .map_err(APIError::from)?;
    Ok(web::Json(r))
}

#[actix_web::get("/v1/models")]
async fn models() -> Result<web::Json<openai::responses::List<openai::responses::Model>>, APIError>
{
    Ok(web::Json(openai::responses::List::new(vec![
        openai::responses::Model {
            object: "model",
            id: "test".to_string(),
            created: 1686935002,
            owned_by: "you".to_string(),
        },
    ])))
}

pub enum InferenceReq {
    AddRequest(AddRequest),
}

type InferenceResult = Result<RequestOutput>;

pub struct InferenceWorker {
    req_sender: Sender<InferenceReq>,
    running: HashMap<String, Sender<InferenceResult>>,
}

impl InferenceWorker {
    pub fn new() -> (Self, Receiver<InferenceReq>) {
        let (tx, rx) = channel(128);
        let r = Self {
            req_sender: tx,
            running: HashMap::new(),
        };
        (r, rx)
    }
    pub fn add_request(&mut self, req: AddRequest) -> Result<Receiver<InferenceResult>> {
        let (tx, rx) = channel(128);
        let rid = req.request_id.clone();
        self.req_sender.try_send(InferenceReq::AddRequest(req))?;
        self.running.insert(rid, tx);
        Ok(rx)
    }
}

fn inference_loop(
    handle: Arc<Mutex<InferenceWorker>>,
    mut engine: RllmEngine,
    mut recv: Receiver<InferenceReq>,
) {
    loop {
        loop {
            let req = if engine.num_pending_requests() > 0 {
                recv.try_recv()
            } else {
                Ok(recv.blocking_recv().unwrap())
            };
            match req {
                Ok(InferenceReq::AddRequest(req)) => {
                    let id = req.request_id.clone();
                    match engine.queue_request(req) {
                        Ok(_) => {}
                        Err(e) => {
                            let tx = handle.lock().unwrap().running.remove(&id).unwrap();
                            if let Err(e) = tx.try_send(Err(e)) {
                                log::warn!("failed to send error to client {id}: {e}");
                            }
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => panic!(),
            }
        }

        let outputs = engine.step().expect("run_model() failed");

        {
            let running = &mut handle.lock().unwrap().running;
            for outp in outputs {
                let id = outp.request_id.clone();
                let tx = if outp.is_final {
                    running.remove(&id)
                } else {
                    running.get(&id).cloned()
                };

                match tx {
                    Some(tx) => {
                        if let Err(e) = tx.try_send(Ok(outp)) {
                            log::warn!("failed to send output to client {id}: {e}");
                            engine.abort_request(&id);
                        }
                    }
                    None => {
                        if id == "warmup" {
                            if outp.is_final {
                                let text = engine.seq_output_text(&outp.seq_outputs[0]).unwrap();
                                log::info!("warmup done: {text:?}");
                            }
                        } else {
                            log::warn!("output for unknown request {id}");
                            engine.abort_request(&id);
                        }
                    }
                }
            }
        }
    }
}

#[actix_web::main]
async fn main() -> () {
    setup_log();

    let mut args = Args::parse();

    if args.model.contains('@') {
        let m = args.model.clone();
        let mut parts = m.split('@');
        args.model = parts.next().unwrap().to_string();
        args.revision = Some(parts.next().unwrap().to_string());
    }

    let loader_args = LoaderArgs {
        model_id: args.model.clone(),
        revision: args.revision.clone(),
        local_weights: args.local_weights.clone(),
        tokenizer: args.tokenizer.clone(),
        ..LoaderArgs::default()
    };

    let (tokenizer, tok_trie) =
        RllmEngine::load_tokenizer(&loader_args).expect("failed to load tokenizer");
    let model_config =
        RllmEngine::load_model_config(&loader_args).expect("failed to load model config");

    let rt_args = rllm::iface::Args {
        aicirt: args.aicirt.clone(),
        tokenizer: args.tokenizer.clone(),
        json_size: args.json_size,
        bin_size: args.bin_size,
        shm_prefix: args.shm_prefix.clone(),
        busy_wait_time: args.busy_wait_time,
    };
    let iface = AiciRtIface::start_aicirt(&rt_args, &tok_trie).expect("failed to start aicirt");

    let (handle, recv) = InferenceWorker::new();
    let handle = Arc::new(Mutex::new(handle));
    let app_data = OpenAIServerData {
        worker: handle.clone(),
        model_config,
        tokenizer: Arc::new(tokenizer),
        tok_trie: Arc::new(tok_trie),
        side_cmd_ch: iface.side_cmd.clone(),
    };
    let app_data = web::Data::new(app_data);
    let handle2 = handle.clone();

    let profile_step = args.profile_step;
    std::thread::spawn(move || {
        let mut engine = RllmEngine::load(loader_args).expect("failed to load model");
        engine.profile_step_no = profile_step;
        engine.set_aicirt(iface);
        engine
            .add_request(
                "warmup".to_string(),
                "The ultimate answer to life,",
                SamplingParams {
                    max_tokens: 10,
                    ..SamplingParams::default()
                },
            )
            .unwrap();
        inference_loop(handle2, engine, recv)
    });

    let host = "127.0.0.1";

    println!("Listening at http://{}:{}", host, args.port);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .service(models)
            .service(completion::completions)
            .configure(|cfg| {
                cfg.app_data(web::PayloadConfig::new(128 * 1024 * 1024))
                    .service(upload_aici_module);
            })
            .app_data(app_data.clone())
    })
    .workers(3)
    .bind((host, args.port))
    .expect("failed to start server (bind)")
    .run()
    .await
    .expect("failed to start server (run)");
}

pub(crate) fn get_unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
