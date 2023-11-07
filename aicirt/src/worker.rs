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
pub struct ProcessHandle<Cmd, Resp> {
    pid: pid_t,
    cmd: IpcSender<Cmd>,
    cmd_resp: IpcReceiver<Resp>,
    // query: Option<IpcReceiver<Query>>,
    // query_resp: IpcSender<QueryResp>,
}

type ForkerHandle = ProcessHandle<ForkerCmd, ForkerResp>;
type SeqGroupHandle = ProcessHandle<SeqGroupCmd, SeqGroupResp>;
type SeqHandle = ProcessHandle<SeqCmd, SeqResp>;

#[derive(Serialize, Deserialize, Debug)]
struct ForkerCmd {
    id: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ForkerResp(SeqGroupHandle);

#[derive(Serialize, Deserialize, Debug)]
enum SeqCmd {}

#[derive(Serialize, Deserialize, Debug)]
enum SeqResp {}

#[derive(Serialize, Deserialize, Debug)]
enum SeqGroupCmd {
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
    SetId {
        inst_id: ModuleInstId,
    },
}

#[derive(Serialize, Deserialize, Debug)]
enum SeqGroupResp {
    Fork { handle: SeqHandle },
    Ok {},
    Error { msg: String },
}

impl<Cmd, Resp> ProcessHandle<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    fn send_cmd(&self, cmd: Cmd) -> Result<Resp> {
        self.cmd.send(cmd)?;
        Ok(self.cmd_resp.recv()?)
    }
}

impl SeqGroupHandle {
    fn fork(&self, inst_id: ModuleInstId) -> Result<ProcessHandle<SeqCmd, SeqResp>> {
        match self.send_cmd(SeqGroupCmd::Fork { inst_id })? {
            SeqGroupResp::Fork { handle: info } => match info.cmd_resp.recv().unwrap() {
                SeqGroupResp::Ok {} => Ok(info),
                r => Err(anyhow!("unexpected response (child) {r:?}")),
            },
            r => Err(anyhow!("unexpected response {r:?}")),
        }
    }

    fn send_cmd_expect_ok(&self, cmd: SeqGroupCmd) -> Result<()> {
        match self.send_cmd(cmd)? {
            SeqGroupResp::Ok {} => Ok(()),
            r => Err(anyhow!("unexpected response {r:?}")),
        }
    }
}

struct SeqCtx {
    cmd: IpcReceiver<SeqCmd>,
    cmd_resp: IpcSender<SeqResp>,
    wasm_ctx: WasmContext,
    query: ProcessHandle<Query, QueryResp>,
    inst_id: ModuleInstId,
    modinst: ModuleInstance,
}

fn ok() -> Result<SeqGroupResp> {
    Ok(SeqGroupResp::Ok {})
}

struct SeqHandleRes {
    handle: SeqHandle,
    query: IpcReceiver<Query>,
    query_resp: IpcSender<QueryResp>,
}

impl SeqCtx {
    fn dispatch_loop(&mut self) -> ! {
        loop {}
    }

    fn spawn(modinst: ModuleInstance, wasm_ctx: WasmContext) -> SeqHandleRes {
        let (cmd0, cmd1) = ipc::channel().unwrap();
        let (cmd_resp0, cmd_resp1) = ipc::channel().unwrap();

        let (query0, query1) = ipc::channel().unwrap();
        let (query_resp0, query_resp1) = ipc::channel().unwrap();

        let mut ctx = SeqCtx {
            cmd: cmd1,
            cmd_resp: cmd_resp0,
            query: ProcessHandle {
                pid: 0,
                cmd: query0,
                cmd_resp: query_resp1,
            },
            wasm_ctx,
            inst_id: 0,
            modinst,
        };

        let pid = unsafe { libc::fork() };
        if pid == 0 {
            ctx.dispatch_loop()
        } else {
            let handle = ProcessHandle {
                pid,
                cmd: cmd0,
                cmd_resp: cmd_resp1,
            };
            SeqHandleRes {
                handle,
                query: query1,
                query_resp: query_resp0,
            }
        }
    }
}

impl SeqGroupCtx {
    fn dispatch_one(&mut self, cmd: SeqGroupCmd) -> Result<SeqGroupResp> {
        match cmd {
            SeqGroupCmd::Fork { inst_id } => {
                let (mut wi, ctx) = new_channels(self.wasm_ctx.clone());
                let pid = unsafe { libc::fork() };
                if pid == 0 {
                    *self = ctx;
                    self.inst_id = inst_id;
                    if self.modinst.is_some() {
                        self.modinst.as_mut().unwrap().set_id(inst_id);
                    }
                    // note that this is sent over the child channel
                    // we do it this way, so that we come back to dispatch_loop()
                    // and continue in the child with the same stack height as in the parent
                    ok()
                } else {
                    wi.pid = pid;
                    Ok(SeqGroupResp::Fork { handle: wi })
                }
            }
            SeqGroupCmd::Instantiate {
                module_path,
                module_id,
                module_arg,
                prompt_str,
                prompt_toks,
            } => {
                let module = self.wasm_ctx.deserialize_module(module_path).unwrap();
                let _ = module_id;
                let mut inst = ModuleInstance::new(
                    424242,
                    self.wasm_ctx.clone(),
                    module,
                    Arc::new(module_arg),
                )?;
                let prompt_toks = if let Some(t) = prompt_toks {
                    t
                } else {
                    inst.tokenize(prompt_str.as_ref().unwrap())?
                };

                let handle = SeqCtx::spawn(inst, self.wasm_ctx.clone());
                self.workers.insert(handle.handle.pid as u64, handle.handle.clone());
                handle.handle.send_cmd_expect_ok(SeqCmd::SetTokens {
                    tokens: prompt_toks,
                })?;

                // inst.setup(&prompt_toks)?;

                ok()
            }
            SeqGroupCmd::SetId { inst_id } => {
                self.inst_id = inst_id;
                self.modinst.as_mut().unwrap().set_id(inst_id);
                ok()
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
            let resp = match self.dispatch_one(self.cmd.recv().unwrap()) {
                Ok(v) => v,
                Err(e) => SeqGroupResp::Error {
                    msg: format!("{e:?}"),
                },
            };
            self.cmd_resp.send(resp).unwrap();
        }
    }
}


pub struct SeqGroupCtx {
    id: String,
    cmd: IpcReceiver<SeqGroupCmd>,
    cmd_resp: IpcSender<SeqGroupResp>,
    wasm_ctx: WasmContext,
    variables: HashMap<String, Vec<u8>>,
    workers: HashMap<u64, SeqHandle>,
    cb_set: IpcReceiverSet,
}

impl SeqGroupCtx {
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
}

pub struct SeqGroupWorkerHandle {
    handle: SeqGroupHandle,
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
    fork_worker: ForkerHandle,
}

fn forker_dispatcher(
    cmd: IpcReceiver<ForkerCmd>,
    cmd_resp: IpcSender<ForkerResp>,
    wasm_ctx: WasmContext,
) -> ! {
    loop {
        let cmd = cmd.recv().unwrap();

        let (cmd0, cmd1) = ipc::channel().unwrap();
        let (cmd_resp0, cmd_resp1) = ipc::channel().unwrap();

        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let mut ctx = SeqGroupCtx {
                id: cmd.id,
                cmd: cmd1,
                cmd_resp: cmd_resp0,
                wasm_ctx,
                variables: HashMap::new(),
                workers: HashMap::new(),
                cb_set: IpcReceiverSet::new().unwrap(),
            };
            ctx.dispatch_loop()
        } else {
            let fork_worker = ProcessHandle {
                pid,
                cmd: cmd0,
                cmd_resp: cmd_resp1,
            };
            cmd_resp.send(ForkerResp(fork_worker)).unwrap();
        }
    }
}

impl WorkerForker {
    pub fn new(wasm_ctx: WasmContext) -> Self {
        let (cmd0, cmd1) = ipc::channel().unwrap();
        let (cmd_resp0, cmd_resp1) = ipc::channel().unwrap();

        let pid = unsafe { libc::fork() };
        if pid == 0 {
            forker_dispatcher(cmd1, cmd_resp0, wasm_ctx)
        } else {
            let fork_worker = ProcessHandle {
                pid,
                cmd: cmd0,
                cmd_resp: cmd_resp1,
            };
            WorkerForker { fork_worker }
        }
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

        let resp = self.fork_worker.send_cmd(ForkerCmd {
            id: req.req_id.clone(),
        })?;
        let handle = resp.0;
        handle.send_cmd_expect_ok(SeqGroupCmd::Instantiate {
            module_path,
            module_id: req.module_id.clone(),
            module_arg,
            prompt_str,
            prompt_toks,
        })?;
        Ok(SeqGroupWorkerHandle { handle })
    }
}
