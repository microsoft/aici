use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use actix_web::{middleware::Logger, web, App, HttpServer};
use anyhow::Result;
use clap::Parser;

use rllm::{seq::RequestOutput, AddRequest, LoaderArgs, RllmEngine};

mod openai;

use openai::responses::APIError;
use tokio::sync::mpsc::{channel, error::TryRecvError, Receiver, Sender};

use crate::openai::openai_server::completions;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Port to serve on (localhost:port)
    #[arg(long)]
    port: u16,

    /// Set verbose mode (print all requests)
    #[arg(long)]
    verbose: bool,

    /// Huggingface model name
    #[arg(long)]
    model_id: Option<String>,

    /// Huggingface model revision
    #[arg(long)]
    revision: Option<String>,

    /// The folder name that contains safetensor weights and json files
    /// (same structure as huggingface online)
    #[arg(long)]
    local_weights: Option<String>,
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
                        log::warn!("output for unknown request {id}");
                        engine.abort_request(&id);
                    }
                }
            }
        }
    }
}

#[actix_web::main]

async fn main() -> Result<()> {
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.init();

    let args = Args::parse();
    let loader_args = LoaderArgs {
        model_id: args.model_id,
        revision: args.revision,
        local_weights: args.local_weights,
        use_reference: false,
        alt: 0,
    };
    let tokenizer = RllmEngine::load_tokenizer(&loader_args)?;

    let (handle, recv) = InferenceWorker::new();
    let handle = Arc::new(Mutex::new(handle));
    let app_data = openai::OpenAIServerData {
        worker: handle.clone(),
        pipeline_config: openai::PipelineConfig {
            max_model_len: 1024,
        },
        tokenizer: Arc::new(tokenizer),
    };
    let app_data = web::Data::new(app_data);
    let handle2 = handle.clone();

    std::thread::spawn(move || {
        let engine = RllmEngine::load(loader_args).expect("failed to load model");
        inference_loop(handle2, engine, recv)
    });

    let host = "127.0.0.1";

    println!("Listening at http://{}:{}", host, args.port);
    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .service(models)
            .service(completions)
            .app_data(app_data.clone())
    })
    .workers(3)
    .bind((host, args.port))
    .map_err(|e| APIError::new(e.to_string()))?
    .run()
    .await
    .map_err(|e| APIError::new(e.to_string()))?;

    Ok(())
}
