use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

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

pub type JSON = serde_json::Value;

#[derive(Serialize, Deserialize, Debug)]
enum GroupCmd {
    NewChannel {},
    ReadVar { name: String },
    WriteVar { name: String, value: Vec<u8> },
}

#[derive(Serialize, Deserialize, Debug)]
enum GroupResp {
    NewChannel { channel: GroupHandle },
    ReadVar { value: Vec<u8> },
    Ok {},
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessHandle<Cmd, Resp> {
    pid: pid_t,
    cmd: IpcSender<Cmd>,
    cmd_resp: IpcReceiver<Resp>,
}

type ForkerHandle = ProcessHandle<ForkerCmd, ForkerResp>;
type SeqHandle = ProcessHandle<SeqCmd, SeqResp>;
type GroupHandle = ProcessHandle<GroupCmd, GroupResp>;

#[derive(Serialize, Deserialize, Debug)]
struct ForkerCmd {
    id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecOp {
    pub op: AiciOp,
    pub logit_offset: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct ForkerResp(SeqHandle);

#[derive(Serialize, Deserialize, Debug)]
enum SeqCmd {
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
    Exec {
        data: ExecOp,
    },
}

#[derive(Serialize, Deserialize, Debug)]
enum SeqResp {
    Fork { handle: SeqHandle },
    Ok {},
    Exec { data: JSON },
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

impl SeqHandle {
    fn send_cmd_expect_ok(&self, cmd: SeqCmd) -> Result<()> {
        match self.send_cmd(cmd)? {
            SeqResp::Ok {} => Ok(()),
            r => Err(anyhow!("unexpected response (not OK) {r:?}")),
        }
    }
}

fn ok() -> Result<SeqResp> {
    Ok(SeqResp::Ok {})
}

struct SeqHandleRes {
    handle: SeqHandle,
    query: IpcReceiver<GroupCmd>,
    query_resp: IpcSender<GroupResp>,
}

impl SeqCtx {
    fn dispatch_one(&mut self, cmd: SeqCmd) -> Result<SeqResp> {
        match cmd {
            SeqCmd::Fork { inst_id } => {
                let (cmd0, cmd1) = ipc::channel().unwrap();
                let (cmd_resp0, cmd_resp1) = ipc::channel().unwrap();
                let w_pid = unsafe { libc::fork() };

                let query_ch = match self.group_cmd(GroupCmd::NewChannel {}) {
                    GroupResp::NewChannel { channel } => channel,
                    r => return Err(anyhow!("unexpected response (SeqCtx.dispatch) {r:?}")),
                };

                if w_pid == 0 {
                    self.cmd = cmd1;
                    self.cmd_resp = cmd_resp0;
                    self.query = query_ch;
                    self.inst_id = inst_id;
                    self.mutinst().set_id(inst_id);
                    // note that this is sent over the child channel
                    // we do it this way, so that we come back to dispatch_loop()
                    // and continue in the child with the same stack height as in the parent
                    ok()
                } else {
                    let handle = ProcessHandle {
                        pid: w_pid,
                        cmd: cmd0,
                        cmd_resp: cmd_resp1,
                    };
                    Ok(SeqResp::Fork { handle })
                }
            }
            SeqCmd::Instantiate {
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
                self.modinst = Some(inst);
                self.mutinst().setup(&prompt_toks)?;
                ok()
            }
            SeqCmd::SetId { inst_id } => {
                self.inst_id = inst_id;
                self.mutinst().set_id(inst_id);
                ok()
            }
            SeqCmd::Exec { data } => {
                todo!()
            }
        }
    }

    fn mutinst(&mut self) -> &mut ModuleInstance {
        self.modinst.as_mut().unwrap()
    }

    fn group_cmd(&mut self, query: GroupCmd) -> GroupResp {
        self.query.send_cmd(query).unwrap()
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            let resp = match self.dispatch_one(self.cmd.recv().unwrap()) {
                Ok(v) => v,
                Err(e) => SeqResp::Error {
                    msg: format!("{e:?}"),
                },
            };
            self.cmd_resp.send(resp).unwrap();
        }
    }
}

struct GroupCtx {
    variables: HashMap<String, Vec<u8>>,
    workers: HashMap<u64, IpcSender<GroupResp>>,
    cb_set: IpcReceiverSet,
}

struct SeqCtx {
    id: String,
    cmd: IpcReceiver<SeqCmd>,
    cmd_resp: IpcSender<SeqResp>,
    wasm_ctx: WasmContext,
    query: GroupHandle,
    inst_id: ModuleInstId,
    modinst: Option<ModuleInstance>,
}

pub struct SeqWorkerHandle {
    handle: SeqHandle,
}

impl SeqWorkerHandle {
    pub fn free(&self) {
        todo!()
    }

    pub fn kill(&self) {
        todo!()
    }

    pub fn set_id(&self, id: ModuleInstId) -> Result<()> {
        self.handle
            .send_cmd_expect_ok(SeqCmd::SetId { inst_id: id })
    }

    pub fn fork(&self, target_id: ModuleInstId) -> Result<SeqWorkerHandle> {
        match self.handle.send_cmd(SeqCmd::Fork { inst_id: target_id })? {
            SeqResp::Fork { handle: info } => match info.cmd_resp.recv().unwrap() {
                SeqResp::Ok {} => Ok(SeqWorkerHandle { handle: info }),
                r => Err(anyhow!("unexpected response (fork, child) {r:?}")),
            },
            r => Err(anyhow!("unexpected response (fork) {r:?}")),
        }
    }

    pub fn start_exec(&self, data: ExecOp) -> Result<()> {
        self.handle.cmd.send(SeqCmd::Exec { data })?;
        Ok(())
    }

    pub fn check_exec(&self, timeout: Duration) -> Result<Option<JSON>> {
        match self.handle.cmd_resp.try_recv_timeout(timeout) {
            Ok(SeqResp::Exec { data }) => Ok(Some(data)),
            Ok(r) => Err(anyhow!("unexpected response (exec) {r:?}")),
            Err(ipc_channel::ipc::TryRecvError::Empty) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

impl GroupCtx {
    fn add_worker(&mut self, query: IpcReceiver<GroupCmd>, query_resp: IpcSender<GroupResp>) {
        let id = self.cb_set.add(query).unwrap();
        self.workers.insert(id, query_resp);
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            for ent in self.cb_set.select().unwrap() {
                match ent {
                    ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                        let cmd: GroupCmd = msg.to().unwrap();
                        let resp = match cmd {
                            GroupCmd::NewChannel {} => {
                                let (query0, query1) = ipc::channel().unwrap();
                                let (query_resp0, query_resp1) = ipc::channel().unwrap();
                                self.add_worker(query1, query_resp0);
                                GroupResp::NewChannel {
                                    channel: ProcessHandle {
                                        pid: 0,
                                        cmd: query0,
                                        cmd_resp: query_resp1,
                                    },
                                }
                            }
                            GroupCmd::ReadVar { name } => GroupResp::ReadVar {
                                value: self
                                    .variables
                                    .get(&name)
                                    .map(|x| x.clone())
                                    .unwrap_or_else(Vec::new),
                            },
                            GroupCmd::WriteVar { name, value } => {
                                self.variables.insert(name, value);
                                GroupResp::Ok {}
                            }
                        };
                        self.workers.get(&id).unwrap().send(resp).unwrap()
                    }
                    ipc::IpcSelectionResult::ChannelClosed(id) => {
                        self.workers.remove(&id);
                        if self.workers.len() == 0 {
                            std::process::exit(0);
                        }
                    }
                }
            }
        }
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

        let (query0, query1) = ipc::channel().unwrap();
        let (query_resp0, query_resp1) = ipc::channel().unwrap();

        let grp_pid = unsafe { libc::fork() };
        if grp_pid == 0 {
            let mut grp_ctx = GroupCtx {
                variables: HashMap::new(),
                workers: HashMap::new(),
                cb_set: IpcReceiverSet::new().unwrap(),
            };
            grp_ctx.add_worker(query1, query_resp0);
            grp_ctx.dispatch_loop()
        } else {
            let w_pid = unsafe { libc::fork() };
            if w_pid == 0 {
                let mut w_ctx = SeqCtx {
                    id: cmd.id,
                    cmd: cmd1,
                    cmd_resp: cmd_resp0,
                    wasm_ctx,
                    query: ProcessHandle {
                        pid: grp_pid,
                        cmd: query0,
                        cmd_resp: query_resp1,
                    },
                    inst_id: 424242,
                    modinst: None,
                };
                w_ctx.dispatch_loop()
            } else {
                let fork_worker = ProcessHandle {
                    pid: w_pid,
                    cmd: cmd0,
                    cmd_resp: cmd_resp1,
                };
                cmd_resp.send(ForkerResp(fork_worker)).unwrap();
            }
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
    ) -> Result<SeqWorkerHandle> {
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
        handle.send_cmd_expect_ok(SeqCmd::Instantiate {
            module_path,
            module_id: req.module_id.clone(),
            module_arg,
            prompt_str,
            prompt_toks,
        })?;
        Ok(SeqWorkerHandle { handle })
    }
}
