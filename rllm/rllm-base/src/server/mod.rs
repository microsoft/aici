use crate::{
    config::{ModelMeta, SamplingParams},
    iface::{kill_self, AiciRtIface, AsyncCmdChannel},
    seq::RequestOutput,
    util::apply_settings,
    AddRequest, HashMap, LoaderArgs, ModelExec, RllmEngine,
};
use actix_web::{middleware::Logger, web, App, HttpServer};
use aici_abi::toktree::TokTrie;
use aicirt::{
    api::{AuthInfo, GetTagsResp, MkModuleReq, MkModuleResp, SetTagsReq},
    bintokens::{guess_tokenizer, list_tokenizers},
    set_max_priority, UserError,
};
use anyhow::{bail, Result};
use base64::Engine;
use clap::Args;
use std::{
    fmt::Display,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::sync::mpsc::{channel, error::TryRecvError, Receiver, Sender};

mod api;
mod completion;
mod openai;

#[derive(Debug)]
pub struct APIError {
    code: actix_web::http::StatusCode,
    msg: String,
}

impl From<anyhow::Error> for APIError {
    fn from(e: anyhow::Error) -> Self {
        Self::from_anyhow(e)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for APIError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::from_anyhow(anyhow::anyhow!(e))
    }
}

impl std::error::Error for APIError {}
impl actix_web::error::ResponseError for APIError {
    fn status_code(&self) -> actix_web::http::StatusCode {
        self.code
    }
}

impl Display for APIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "APIError: {}", self.msg)
    }
}

impl APIError {
    pub fn new(data: String) -> Self {
        Self {
            code: actix_web::http::StatusCode::BAD_REQUEST,
            msg: data,
        }
    }

    pub fn new_str(data: &str) -> Self {
        Self::new(data.to_string())
    }

    pub fn from_anyhow(value: anyhow::Error) -> Self {
        if UserError::is_self(&value) {
            log::info!("UserError: {value}");
            Self {
                code: actix_web::http::StatusCode::BAD_REQUEST,
                msg: format!("{value}"),
            }
        } else {
            log::warn!("APIError: {value:?}");
            Self {
                code: actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
                msg: format!("{value:?}"),
            }
        }
    }

    pub fn just_msg(value: anyhow::Error) -> Self {
        log::info!("Error: {value}");
        Self {
            code: actix_web::http::StatusCode::BAD_REQUEST,
            msg: format!("{value}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerStats {
    pub num_requests: usize,
    pub num_tokens: usize,
    pub start_time: Instant,
}

impl Display for ServerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "requests: {}; tokens: {}; uptime: {:?}",
            self.num_requests,
            self.num_tokens,
            self.start_time.elapsed()
        )
    }
}

#[derive(Clone)]
pub struct AiciServerData {
    pub worker: Arc<Mutex<InferenceWorker>>,
    pub model_meta: ModelMeta,
    pub tokenizer: Arc<tokenizers::Tokenizer>,
    pub tok_trie: Arc<TokTrie>,
    pub side_cmd_ch: AsyncCmdChannel,
    pub stats: Arc<Mutex<ServerStats>>,
}

#[derive(Args, Debug)]
pub struct RllmCliArgs {
    /// Set engine setting (see below or in --help for list)
    #[arg(long, short, name = "NAME=VALUE")]
    pub setting: Vec<String>,

    /// Set log level (ex: "info,rllm=trace,aicirt=debug", "warn", "error")
    #[arg(long)]
    pub log: Option<String>,

    /// HuggingFace model name, URL or path starting with "./"
    #[arg(short, long, help_heading = "Model")]
    pub model: String,

    /// HuggingFace model revision; --model foo/bar@revision is also possible
    #[arg(long, help_heading = "Model")]
    pub revision: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as HuggingFace online)
    #[arg(long, help_heading = "Model")]
    pub local_weights: Option<String>,

    /// Tokenizer to use (see below or in --help for list)
    #[arg(short, long, help_heading = "Model")]
    pub tokenizer: Option<String>,

    /// Port to serve on (localhost:port)
    #[arg(long, default_value_t = 4242, help_heading = "Server")]
    pub port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long, default_value_t = false, help_heading = "Server")]
    pub verbose: bool,

    /// Enable daemon mode (log timestamps)
    #[arg(long, default_value_t = false, help_heading = "Server")]
    pub daemon: bool,

    /// Path to the aicirt binary.
    #[arg(long, help_heading = "AICI settings")]
    pub aicirt: Option<String>,

    /// Size of JSON comm buffer in megabytes
    #[arg(long, default_value = "128", help_heading = "AICI settings")]
    pub json_size: usize,

    /// Size of binary comm buffer in megabytes
    #[arg(long, default_value = "32", help_heading = "AICI settings")]
    pub bin_size: usize,

    /// How many milliseconds to spin-wait for a message over IPC and SHM.
    #[arg(long, default_value = "200", help_heading = "AICI settings")]
    pub busy_wait_time: u64,

    /// Shm/semaphore name prefix; default /aici-PORT-
    #[arg(long, help_heading = "AICI settings")]
    pub shm_prefix: Option<String>,

    /// Specify test-cases (expected/*/*.safetensors)
    #[arg(long, help_heading = "Development")]
    pub test: Vec<String>,

    /// Specify warm-up request (expected/*/*.safetensors or "off")
    #[arg(long, short, help_heading = "Development")]
    pub warmup: Option<String>,

    /// Exit after processing warmup request
    #[arg(long, default_value_t = false, help_heading = "Development")]
    pub warmup_only: bool,

    // these are copied from command-specific parsers
    #[arg(skip)]
    pub file: Option<String>,
}

#[actix_web::get("/v1/controllers/tags")]
async fn get_controllers_tags(
    req: actix_web::HttpRequest,
    data: web::Data<AiciServerData>,
) -> Result<web::Json<GetTagsResp>, APIError> {
    let r = data
        .side_cmd_ch
        .get_tags(auth_info(&req))
        .await
        .map_err(APIError::just_msg)?;
    Ok(web::Json(r))
}

#[actix_web::post("/v1/controllers/tags")]
async fn tag_controller(
    req: actix_web::HttpRequest,
    data: web::Data<AiciServerData>,
    body: web::Json<SetTagsReq>,
) -> Result<web::Json<GetTagsResp>, APIError> {
    let r = data
        .side_cmd_ch
        .set_tags(body.0, auth_info(&req))
        .await
        .map_err(APIError::just_msg)?;
    Ok(web::Json(r))
}

#[actix_web::post("/v1/controllers")]
async fn upload_controller(
    req: actix_web::HttpRequest,
    data: web::Data<AiciServerData>,
    body: web::Bytes,
) -> Result<web::Json<MkModuleResp>, APIError> {
    let binary = base64::engine::general_purpose::STANDARD.encode(body);
    let r = data
        .side_cmd_ch
        .mk_module(MkModuleReq { binary }, auth_info(&req))
        .await
        .map_err(APIError::just_msg)?;
    Ok(web::Json(r))
}

#[actix_web::get("/v1/models")]
async fn models(
    data: web::Data<AiciServerData>,
) -> Result<web::Json<openai::responses::List<openai::responses::Model>>, APIError> {
    let id = data.model_meta.id.clone();
    Ok(web::Json(openai::responses::List::new(vec![
        openai::responses::Model {
            object: "model",
            id,
            created: 946810800,
            owned_by: "owner".to_string(),
        },
    ])))
}

pub fn auth_info(req: &actix_web::HttpRequest) -> AuthInfo {
    // we default to localhost/admin when no headers given
    let user = req
        .headers()
        .get("x-user-id")
        .map_or("localhost", |v| v.to_str().unwrap_or("(invalid header)"));
    let role = req
        .headers()
        .get("x-user-role")
        .map_or("admin", |v| v.to_str().unwrap_or("(invalid header)"));
    let is_admin = role == "admin";
    AuthInfo {
        user: user.to_string(),
        is_admin,
    }
}

#[actix_web::get("/ws-http-tunnel/info")]
async fn tunnel_info(
    req: actix_web::HttpRequest,
    data: web::Data<AiciServerData>,
) -> Result<web::Json<serde_json::Value>, APIError> {
    let name = req
        .headers()
        .get("x-user-id")
        .map_or("(no header)", |v| v.to_str().unwrap_or("(invalid header)"));
    log::info!("user: {:?}", name);
    let url = "https://github.com/microsoft/aici/blob/main/docs/proxy.md";
    let model = &data.model_meta.id;
    let stats = data.stats.lock().unwrap().clone();
    let msg = format!(
        r#"
Model: {model}
Stats: {stats}

More info at: {url}"#
    );
    Ok(web::Json(serde_json::json!({
        "msg": msg,
        "connection_string": "AICI_API_BASE=\"{website}/v1/#key={key}\""
    })))
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
            running: HashMap::default(),
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

fn inference_loop<ME: ModelExec>(
    handle: Arc<Mutex<InferenceWorker>>,
    mut engine: RllmEngine<ME>,
    mut recv: Receiver<InferenceReq>,
    stats: Arc<Mutex<ServerStats>>,
    warmup_only: bool,
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
                        Ok(_) => {
                            let mut stats = stats.lock().unwrap();
                            stats.num_requests += 1;
                        }
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
            let mut stats = stats.lock().unwrap();
            stats.num_tokens += 1;
        }

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
                                if warmup_only {
                                    log::info!("warmup done; exiting");
                                    kill_self();
                                }
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

fn run_tests<ME: ModelExec>(
    args: &RllmCliArgs,
    loader_args: LoaderArgs,
    model_args: ME::ModelLoaderArgs,
) {
    let mut engine = ME::load_rllm_engine(loader_args, model_args).expect("failed to load model");
    let mut tests = args.test.clone();

    while tests.len() > 0 || engine.num_pending_requests() > 0 {
        if let Some(ref t) = tests.pop() {
            let exp = crate::ExpectedGeneration::load(&std::path::PathBuf::from(t))
                .expect("can't load test");
            log::info!(
                "test {t}: {} tokens; {} logits",
                exp.output.len(),
                exp.output[0].logits.len()
            );
            if exp.output.len() > 11 {
                let mut exp2 = exp.clone();
                engine.add_expected_generation(exp, None).unwrap();
                // add a few tokens in one go to test
                exp2.output[6].ff_section_len = 4;
                engine.add_expected_generation(exp2, None).unwrap();
            } else {
                engine.add_expected_generation(exp, None).unwrap();
            }
        }

        engine.step().expect("test step failed");
    }

    if engine.num_errors > 0 {
        log::error!("there were {} errors", engine.num_errors);
        println!("there were {} errors", engine.num_errors);
        std::process::exit(102);
    }
}

fn spawn_inference_loop<ME: ModelExec>(
    args: &RllmCliArgs,
    loader_args: LoaderArgs,
    model_args: ME::ModelLoaderArgs,
    iface: AiciRtIface,
    stats: Arc<Mutex<ServerStats>>,
) -> Arc<Mutex<InferenceWorker>> {
    let (handle, recv) = InferenceWorker::new();
    let handle_res = Arc::new(Mutex::new(handle));
    let handle = handle_res.clone();

    let warmup = args.warmup.clone();
    let warmup_only = args.warmup_only.clone();

    std::thread::spawn(move || {
        set_max_priority();
        let mut engine =
            ME::load_rllm_engine(loader_args, model_args).expect("failed to load model");
        engine.set_aicirt(iface);
        let wid = "warmup".to_string();
        match warmup {
            Some(w) if w == "off" => {}
            Some(w) => {
                let exp = crate::ExpectedGeneration::load(&std::path::PathBuf::from(&w))
                    .expect("can't load warmup");
                log::info!(
                    "warmup {w}: {} tokens; {} logits",
                    exp.output.len(),
                    exp.output[0].logits.len()
                );
                engine.add_expected_generation(exp, Some(wid)).unwrap();
            }
            _ => {
                engine
                    .add_request(
                        wid,
                        "The ultimate answer to life,",
                        SamplingParams {
                            max_tokens: 10,
                            ..SamplingParams::default()
                        },
                    )
                    .unwrap();
            }
        }
        inference_loop(handle, engine, recv, stats, warmup_only)
    });

    handle_res
}

fn strip_suffix(sep: &str, s: &mut String) -> Option<String> {
    let mut parts = s.splitn(2, sep);
    let core = parts.next().unwrap().to_string();
    let suff = parts.next().map(|s| s.to_string());
    *s = core;
    suff
}

fn url_decode(encoded_str: &str) -> String {
    percent_encoding::percent_decode_str(encoded_str)
        .decode_utf8()
        .unwrap()
        .to_string()
}

fn guess_aicirt() -> Result<String> {
    let mut path = std::env::current_exe()?;
    path.pop();
    path.push("aicirt");
    if path.to_str().is_some() && path.exists() {
        Ok(path.to_str().unwrap().to_string())
    } else {
        bail!("can't find aicirt binary (tried {:?})", path)
    }
}

// #[actix_web::main]
pub async fn server_main<ME: ModelExec>(
    mut args: RllmCliArgs,
    mut model_args: ME::ModelLoaderArgs,
) -> () {
    // we setenv, so that aicirt process also gets it
    match &args.log {
        Some(v) => std::env::set_var("RUST_LOG", v),
        None => {}
    }
    aicirt::init_log(if args.daemon {
        aicirt::LogMode::Daemon
    } else {
        aicirt::LogMode::Normal
    })
    .expect("Failed to initialize log");

    match apply_settings(&args.setting) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(101);
        }
    }

    let hf = "https://huggingface.co/";
    if args.model.starts_with(hf) {
        args.model = args.model[hf.len()..].to_string();

        if let Some(url_rev) = strip_suffix("/tree/", &mut args.model) {
            args.revision = Some(url_decode(&url_rev));
        }

        if let Some(mut blob_path) = strip_suffix("/blob/", &mut args.model) {
            if let Some(url_rev) = strip_suffix("/", &mut blob_path) {
                args.file = Some(url_decode(&url_rev));
            }
            args.revision = Some(url_decode(&blob_path));
        }
    }

    if let Some(file) = strip_suffix("::", &mut args.model) {
        args.file = Some(file);
    }

    if let Some(rev) = strip_suffix("@", &mut args.model) {
        args.revision = Some(rev);
    }

    if args.model.starts_with(".") {
        args.local_weights = Some(args.model.clone());
    }

    let mut loader_args = LoaderArgs::default();
    loader_args.model_id = args.model.clone();
    loader_args.revision = args.revision.clone();
    loader_args.local_weights = args.local_weights.clone();
    loader_args.file = args.file.clone();

    match &args.tokenizer {
        Some(v) => {
            log::info!("explicit tokenizer: {}", v);
            loader_args.tokenizer = v.clone();
        }
        None => match guess_tokenizer(&loader_args.model_id) {
            Some(v) => {
                log::info!("guessed tokenizer: {}", v);
                loader_args.tokenizer = v;
            }
            None => {
                eprintln!("can't guess tokenizer from {}", loader_args.model_id);
                eprintln!("{}", list_tokenizers());
                std::process::exit(10);
            }
        },
    }

    if args.test.len() > 0 {
        run_tests::<ME>(&args, loader_args, model_args);
        return;
    }

    let (tokenizer, tok_trie) =
        RllmEngine::<ME>::load_tokenizer(&mut loader_args).expect("failed to load tokenizer");

    // make sure we try to load the model before spawning inference thread
    // otherwise, if the model doesn't exist, the inference thread will panic and things get messy
    let (model_meta, _model_config) = ME::load_model_config(&mut loader_args, &mut model_args)
        .expect("failed to load model config");

    let aicirt = match &args.aicirt {
        Some(v) => v.clone(),
        None => match guess_aicirt() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("can't find aicirt; specify with --aicirt=PATH\n{e}");
                std::process::exit(10);
            }
        },
    };

    let shm_prefix = match &args.shm_prefix {
        Some(v) => {
            if v.starts_with("/") {
                v.clone()
            } else {
                format!("/{}", v)
            }
        }
        None => format!("/aici-{}-", args.port),
    };

    let rt_args = crate::iface::Args {
        aicirt,
        tokenizer: loader_args.tokenizer.clone(),
        json_size: args.json_size,
        bin_size: args.bin_size,
        shm_prefix,
        busy_wait_time: args.busy_wait_time,
    };
    let stats = Arc::new(Mutex::new(ServerStats {
        num_requests: 0,
        num_tokens: 0,
        start_time: Instant::now(),
    }));
    let iface = AiciRtIface::start_aicirt(&rt_args, &tok_trie).expect("failed to start aicirt");
    let side_cmd_ch = iface.side_cmd.clone();
    let handle = spawn_inference_loop::<ME>(&args, loader_args, model_args, iface, stats.clone());

    let app_data = AiciServerData {
        worker: handle.clone(),
        model_meta,
        tokenizer: Arc::new(tokenizer),
        tok_trie: Arc::new(tok_trie),
        side_cmd_ch,
        stats,
    };
    let app_data = web::Data::new(app_data);
    let host = "127.0.0.1";

    println!("Listening at http://{}:{}", host, args.port);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .service(models)
            .service(tunnel_info)
            .service(completion::run_controller)
            .service(get_controllers_tags)
            .service(tag_controller)
            .configure(|cfg| {
                cfg.app_data(web::PayloadConfig::new(128 * 1024 * 1024))
                    .service(upload_controller);
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
