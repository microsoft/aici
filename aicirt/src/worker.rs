use std::{collections::HashMap, path::PathBuf, sync::Arc};

use aici_abi::TokenId;
use anyhow::{anyhow, Result};
use ipc_channel::ipc::{self, IpcReceiver, IpcReceiverSet, IpcSender};
use libc::pid_t;
use serde::{Deserialize, Serialize};

use crate::{
    hostimpl::ModuleInstId,
    moduleinstance::{ModuleInstance, WasmContext},
    AiciOp, InstantiateReq,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct Query {}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryResp {}

#[derive(Serialize, Deserialize, Debug)]
pub struct WorkerHandle {
    pid: pid_t,
    cmd: IpcSender<WorkerCmd>,
    cmd_resp: IpcReceiver<WorkerResp>,
    query: Option<IpcReceiver<Query>>,
    query_resp: IpcSender<QueryResp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum WorkerCmd {
    Instantiate {
        module_path: PathBuf,
        module_id: String,
        module_arg: String,
        prompt_str: Option<String>,
        prompt_toks: Option<Vec<TokenId>>,
    },
    Fork {
        inst_id: ModuleInstId,
    },
}

#[derive(Serialize, Deserialize, Debug)]
enum WorkerResp {
    Fork { info: WorkerHandle },
    Ok {},
}

impl WorkerHandle {
    fn send_cmd(&self, cmd: WorkerCmd) -> WorkerResp {
        self.cmd.send(cmd).unwrap();
        self.cmd_resp.recv().unwrap()
    }
}

struct WorkerCtx {
    // TODO move channels into substruct
    cmd: IpcReceiver<WorkerCmd>,
    cmd_resp: IpcSender<WorkerResp>,
    query: IpcSender<Query>,
    query_resp: IpcReceiver<QueryResp>,
    inst_id: ModuleInstId,
    wasm_ctx: WasmContext,
}

impl WorkerCtx {
    fn dispatch_one(&mut self, cmd: WorkerCmd) -> WorkerResp {
        match cmd {
            WorkerCmd::Instantiate {
                module_path,
                module_id,
                module_arg,
                prompt_str,
                prompt_toks,
            } => {
                let module = self.wasm_ctx.deserialize_module(module_path).unwrap();
                let inst = ModuleInstance::new(
                    self.inst_id,
                    self.wasm_ctx.clone(),
                    module,
                    Arc::new(module_arg),
                );
                WorkerResp::Ok {}
            }
            WorkerCmd::Fork { inst_id } => {
                let (mut wi, ctx) = new_channels(self.wasm_ctx.clone());
                let pid = unsafe { libc::fork() };
                if pid == 0 {
                    *self = ctx;
                    self.inst_id = inst_id;
                    // note that this is sent over the child channel
                    // we do it this way, so that we come back to dispatch_loop()
                    // and continue in the child with the same stack height as in the parent
                    WorkerResp::Ok {}
                } else {
                    wi.pid = pid;
                    WorkerResp::Fork { info: wi }
                }
            }
        }
    }

    #[allow(dead_code)]
    fn query(&mut self, query: Query) -> QueryResp {
        self.query.send(query).unwrap();
        self.query_resp.recv().unwrap()
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            let resp = self.dispatch_one(self.cmd.recv().unwrap());
            self.cmd_resp.send(resp).unwrap();
        }
    }
}

fn new_channels(wasm_ctx: WasmContext) -> (WorkerHandle, WorkerCtx) {
    let (cmd0, cmd1) = ipc::channel().unwrap();
    let (cmd_resp0, cmd_resp1) = ipc::channel().unwrap();

    let (query0, query1) = ipc::channel().unwrap();
    let (query_resp0, query_resp1) = ipc::channel().unwrap();

    let wi = WorkerHandle {
        pid: 0,
        cmd: cmd0,
        cmd_resp: cmd_resp1,
        query: Some(query1),
        query_resp: query_resp0,
    };

    let ctx = WorkerCtx {
        cmd: cmd1,
        cmd_resp: cmd_resp0,
        query: query0,
        query_resp: query_resp1,
        wasm_ctx,
        inst_id: 0,
    };

    (wi, ctx)
}

pub struct SeqGroupWorkerCtx {
    wasm_ctx: WasmContext,
    variables: HashMap<String, Vec<u8>>,
    workers: HashMap<u64, WorkerHandle>,
    cb_set: IpcReceiverSet,
}

impl SeqGroupWorkerCtx {
    pub fn new(wasm_ctx: WasmContext) -> Self {
        let mgr = SeqGroupWorkerCtx {
            wasm_ctx,
            variables: HashMap::new(),
            workers: HashMap::new(),
            cb_set: IpcReceiverSet::new().unwrap(),
        };
        mgr
    }

    pub fn query_dispatcher(&mut self) -> () {
        loop {
            for ent in self.cb_set.select().unwrap() {
                match ent {
                    ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                        let worker = self.workers.get(&id).unwrap();
                        let _q: Query = msg.to().unwrap();
                        worker.query_resp.send(QueryResp {}).unwrap();
                    }
                    ipc::IpcSelectionResult::ChannelClosed(id) => {
                        self.workers.remove(&id);
                    }
                }
            }
        }
    }

    // pub fn fork(&mut self) -> () {
    //     match self.fork_worker.send_cmd(WorkerCmd::Fork {}) {
    //         WorkerResp::Fork { mut info } => {
    //             match info.cmd_resp.recv().unwrap() {
    //                 WorkerResp::Ok {} => {}
    //                 r => panic!("unexpected response (child) {r:?}"),
    //             }
    //             let q = std::mem::replace(&mut info.query, None).unwrap();
    //             let id = self.cb_set.add(q).unwrap();
    //             self.workers.insert(id, info);
    //         }
    //         r => panic!("unexpected response {r:?}"),
    //     }
    // }
}

pub struct SeqGroupWorkerHandle {
    handle: WorkerHandle,
}

impl SeqGroupWorkerHandle {
    pub fn set_id(&self, id: ModuleInstId) {}
    pub fn create_clone(&self, clone_id: ModuleInstId, target_id: ModuleInstId) {}

    pub fn start_exec(&self, ops: Vec<(AiciOp, usize)>) {}
    pub fn finish_exec(&self) -> Vec<(ModuleInstId, serde_json::Value)> {
        vec![]
    }
}

pub struct WorkerForker {
    fork_worker: WorkerHandle,
}

impl WorkerForker {
    pub fn new(wasm_ctx: WasmContext) -> Self {
        let (mut fork_worker, mut ctx) = new_channels(wasm_ctx);
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            ctx.dispatch_loop()
        } else {
            fork_worker.pid = pid;
        }
        WorkerForker { fork_worker }
    }

    pub fn instantiate(
        &self,
        req: InstantiateReq,
        module_path: PathBuf,
    ) -> Result<SeqGroupWorkerHandle> {
        let module_arg = match req.module_arg.as_str() {
            Some(a) => a.to_string(),
            None => serde_json::to_string(&req.module_arg)?,
        };

        let (prompt_str, prompt_toks) = if req.prompt.is_string() {
            (Some(req.prompt.as_str().unwrap().to_string()), None)
        } else {
            (
                None,
                Some(
                    req.prompt
                        .as_array()
                        .ok_or(anyhow!("expecting string or int array as prompt"))?
                        .iter()
                        .map(|x| -> Result<u32> {
                            x.as_u64()
                                .ok_or(anyhow!("expecting number as token"))?
                                .try_into()
                                .map_err(|e: std::num::TryFromIntError| anyhow!(e))
                        })
                        .collect::<Result<Vec<u32>>>()?,
                ),
            )
        };

        let resp = self.fork_worker.send_cmd(WorkerCmd::Instantiate {
            module_path,
            module_id: req.module_id.clone(),
            module_arg,
            prompt_str,
            prompt_toks,
        });
        match resp {
            WorkerResp::Fork { info } => Ok(SeqGroupWorkerHandle { handle: info }),
            r => Err(anyhow!("unexpected response {r:?}")),
        }
    }
}
