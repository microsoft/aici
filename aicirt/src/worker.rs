use std::fmt::Debug;
use std::rc::Rc;
use std::{collections::HashMap, path::PathBuf, time::Duration};

use aici_abi::host::{StorageCmd, StorageOp, StorageResp};
use aici_abi::{PreProcessArg, ProcessArg, TokenId};
use anyhow::{anyhow, Result};
use ipc_channel::ipc::{self, IpcOneShotServer, IpcReceiver, IpcReceiverSet, IpcSender};
use libc::pid_t;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::bench::{TimerRef, TimerSet};
use crate::with_timer;
use crate::{
    hostimpl::{AiciLimits, ModuleInstId},
    moduleinstance::{ModuleInstance, WasmContext},
    shm::Shm,
    InstantiateReq,
};

pub type JSON = serde_json::Value;

const QUICK_OP_MS: u64 = 10;

#[derive(Serialize, Deserialize, Debug)]
pub enum GroupCmd {
    NewChannel {},
    StorageCmd { cmd: StorageCmd },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum GroupResp {
    NewChannel { channel: GroupHandle },
    StorageResp { resp: StorageResp },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessHandle<Cmd, Resp> {
    pid: pid_t,
    cmd: IpcSender<Cmd>,
    cmd_resp: IpcReceiver<Resp>,
}

pub enum ForkResult<Cmd, Resp> {
    Parent {
        handle: ProcessHandle<Cmd, Resp>,
    },
    Child {
        cmd: IpcReceiver<Cmd>,
        cmd_resp: IpcSender<Resp>,
    },
}

pub fn fork_child<Cmd, Resp>() -> Result<ForkResult<Cmd, Resp>>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    // see https://github.com/servo/ipc-channel/issues/248#issuecomment-559617730
    // specifically cross_process_embedded_senders_fork() in https://github.com/servo/ipc-channel/blob/master/src/test.rs#L259

    let (server, server_name) = IpcOneShotServer::new()?;

    let pid = unsafe { libc::fork() };

    if pid < 0 {
        return Err(anyhow!("fork failed"));
    }

    if pid == 0 {
        let (other_cmd, cmd) = ipc::channel().unwrap();
        let (cmd_resp, other_resp) = ipc::channel().unwrap();

        let tmp = IpcSender::connect(server_name).unwrap();
        tmp.send((other_cmd, other_resp)).unwrap();

        // don't try to free it in the child
        std::mem::forget(server);

        Ok(ForkResult::Child { cmd, cmd_resp })
    } else {
        let (_, (cmd, cmd_resp)) = server.accept().unwrap();

        Ok(ForkResult::Parent {
            handle: ProcessHandle { pid, cmd, cmd_resp },
        })
    }
}

type ForkerHandle = ProcessHandle<ForkerCmd, ForkerResp>;
type SeqHandle = ProcessHandle<SeqCmd, SeqResp>;
pub type GroupHandle = ProcessHandle<GroupCmd, GroupResp>;

#[derive(Serialize, Deserialize, Debug)]
struct ForkerCmd {
    id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RtProcessArg {
    pub op: ProcessArg,
    pub logit_offset: usize,
    pub logit_size: usize, // bytes
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RtPreProcessArg {
    pub op: PreProcessArg,
    pub max_context_size: usize, // elements
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
    SetGroupChannel {
        handle: GroupHandle,
    },
    Fork {
        inst_id: ModuleInstId,
    },
    SetId {
        inst_id: ModuleInstId,
    },
    PreProcess {
        data: RtPreProcessArg,
    },
    Process {
        data: RtProcessArg,
    },
    RunMain {},
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RtPreProcessResult {
    pub json: JSON,
    pub suspend: bool,
    pub attn_masks: Vec<Vec<f32>>,
}

impl RtPreProcessResult {
    pub fn just_json(json: JSON) -> Self {
        RtPreProcessResult {
            json,
            suspend: false,
            attn_masks: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
enum SeqResp {
    Fork {
        handle: SeqHandle,
    },
    Ok {},
    PreProcess {
        json: String,
        suspend: bool,
        attn_masks: Vec<Vec<f32>>,
    },
    Process {
        json: String,
    },
    Error {
        msg: String,
    },
}

impl<Cmd, Resp> ProcessHandle<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize + Debug,
    Resp: for<'d> Deserialize<'d> + Serialize + Debug,
{
    pub fn just_send(&self, cmd: Cmd) -> Result<()> {
        debug!("send {cmd:?}");
        self.cmd.send(cmd)?;
        Ok(())
    }

    pub fn recv(&self) -> Result<Resp> {
        match self.cmd_resp.recv() {
            Ok(r) => {
                debug!("recv {r:?}");
                Ok(r)
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn send_cmd(&self, cmd: Cmd) -> Result<Resp> {
        self.just_send(cmd)?;
        self.recv()
    }

    pub fn kill(&self) -> libc::c_int {
        assert!(self.pid != 0);
        unsafe { libc::kill(self.pid, libc::SIGKILL) }
    }

    fn send_cmd_with_timeout(&self, cmd: Cmd, timeout: Duration) -> Result<Resp> {
        self.just_send(cmd)?;
        self.recv_with_timeout(timeout)
    }

    fn recv_with_timeout(&self, timeout: Duration) -> Result<Resp> {
        match self.cmd_resp.try_recv_timeout(timeout) {
            Ok(r) => {
                debug!("recv t/o {r:?}");
                Ok(r)
            }
            Err(ipc_channel::ipc::TryRecvError::Empty) => {
                Err(anyhow!("timeout ({timeout:?}); pid={}", self.pid))
            }
            Err(e) => {
                // panic!("unexpected error {e:?}");
                Err(e.into())
            }
        }
    }
}

impl SeqHandle {
    fn send_cmd_expect_ok(&self, cmd: SeqCmd, timeout: Duration) -> Result<()> {
        self.just_send(cmd)?;
        match self.recv_with_timeout(timeout) {
            Ok(SeqResp::Ok {}) => Ok(()),
            Ok(SeqResp::Error { msg }) => Err(anyhow!("{}", msg)),
            Ok(r) => Err(anyhow!("unexpected response (not OK) {r:?}")),
            Err(e) => Err(e.into()),
        }
    }
}

fn ok() -> Result<SeqResp> {
    Ok(SeqResp::Ok {})
}

impl SeqCtx {
    fn dispatch_one(&mut self, cmd: SeqCmd) -> Result<SeqResp> {
        match cmd {
            SeqCmd::Fork { inst_id } => {
                match fork_child()? {
                    ForkResult::Parent { handle } => {
                        let group_ch = match self.group_cmd(GroupCmd::NewChannel {}) {
                            GroupResp::NewChannel { channel } => channel,
                            r => {
                                return Err(anyhow!("unexpected response (SeqCtx.dispatch) {r:?}"))
                            }
                        };
                        handle.send_cmd_expect_ok(
                            SeqCmd::SetGroupChannel { handle: group_ch },
                            Duration::from_millis(QUICK_OP_MS),
                        )?;
                        Ok(SeqResp::Fork { handle })
                    }
                    ForkResult::Child { cmd, cmd_resp } => {
                        self.cmd = cmd;
                        self.cmd_resp = cmd_resp;
                        self.inst_id = inst_id;
                        self.mutinst().set_id(inst_id);
                        // note that this is sent over the child channel
                        // we do it this way, so that we come back to dispatch_loop()
                        // and continue in the child with the same stack height as in the parent
                        ok()
                    }
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
                let ch = std::mem::take(&mut self.query);
                let mut inst = ModuleInstance::new(
                    424242,
                    self.wasm_ctx.clone(),
                    module,
                    module_arg,
                    ch.unwrap(),
                )?;
                let prompt_toks = if let Some(t) = prompt_toks {
                    t
                } else {
                    // TODO llama hack
                    let ps = &prompt_str.as_ref().unwrap();
                    let p = if ps.len() == 0 {
                        "<s>".to_string()
                    } else {
                        "<s> ".to_string() + ps
                    };
                    inst.tokenize(&p)?
                };
                self.modinst = Some(inst);
                self.mutinst().setup(prompt_toks)?;
                ok()
            }
            SeqCmd::SetId { inst_id } => {
                self.inst_id = inst_id;
                self.mutinst().set_id(inst_id);
                ok()
            }
            SeqCmd::PreProcess { data } => {
                let res = with_timer!(self.pre_timer, self.mutinst().pre_process(data));
                Ok(SeqResp::PreProcess {
                    json: serde_json::to_string(&res.json)?,
                    suspend: res.suspend,
                    attn_masks: res.attn_masks,
                })
            }
            SeqCmd::Process { data } => {
                let shm = self.shm.clone();
                let res = self.mutinst().process(data, &shm);
                Ok(SeqResp::Process {
                    json: serde_json::to_string(&res)?,
                })
            }
            SeqCmd::RunMain {} => {
                self.mutinst().run_main()?;
                ok()
            }
            SeqCmd::SetGroupChannel { handle } => {
                self.mutinst().set_group_channel(handle);
                ok()
            }
        }
    }

    fn mutinst(&mut self) -> &mut ModuleInstance {
        self.modinst.as_mut().unwrap()
    }

    fn group_cmd(&self, query: GroupCmd) -> GroupResp {
        if let Some(q) = &self.query {
            q.send_cmd(query).unwrap()
        } else {
            self.modinst
                .as_ref()
                .unwrap()
                .group_channel()
                .send_cmd(query)
                .unwrap()
        }
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            let cmd = self.cmd.recv().unwrap();
            debug!("seq recv {cmd:?}");
            let resp = match self.dispatch_one(cmd) {
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
    variables: HashMap<String, (u64, Vec<u8>)>,
    workers: HashMap<u64, IpcSender<GroupResp>>,
    cb_set: IpcReceiverSet,
}

struct SeqCtx {
    #[allow(dead_code)]
    id: String,
    cmd: IpcReceiver<SeqCmd>,
    cmd_resp: IpcSender<SeqResp>,
    wasm_ctx: WasmContext,
    query: Option<GroupHandle>,
    inst_id: ModuleInstId,
    modinst: Option<ModuleInstance>,
    shm: Rc<Shm>,
    pre_timer: TimerRef,
}

pub struct SeqWorkerHandle {
    pub req_id: String,
    handle: SeqHandle,
}

impl Drop for SeqWorkerHandle {
    fn drop(&mut self) {
        self.handle.kill();
    }
}

impl SeqWorkerHandle {
    pub fn set_id(&self, id: ModuleInstId) -> Result<()> {
        self.handle.send_cmd_expect_ok(
            SeqCmd::SetId { inst_id: id },
            Duration::from_millis(QUICK_OP_MS),
        )
    }

    pub fn run_main(&self) -> Result<()> {
        self.handle
            .send_cmd_expect_ok(SeqCmd::RunMain {}, Duration::from_secs(120))
    }

    pub fn fork(&self, target_id: ModuleInstId) -> Result<SeqWorkerHandle> {
        match self.handle.send_cmd_with_timeout(
            SeqCmd::Fork { inst_id: target_id },
            Duration::from_millis(QUICK_OP_MS),
        )? {
            SeqResp::Fork { handle } => {
                let res = SeqWorkerHandle {
                    req_id: self.req_id.clone(),
                    handle,
                };
                match res
                    .handle
                    .recv_with_timeout(Duration::from_millis(QUICK_OP_MS))?
                {
                    SeqResp::Ok {} => Ok(res),
                    r => Err(anyhow!("unexpected response (fork, child) {r:?}")),
                }
            }
            r => Err(anyhow!("unexpected response (fork) {r:?}")),
        }
    }

    pub fn start_pre_process(&self, data: RtPreProcessArg) -> Result<()> {
        self.handle.just_send(SeqCmd::PreProcess { data })?;
        Ok(())
    }

    pub fn start_process(&self, data: RtProcessArg) -> Result<()> {
        self.handle.just_send(SeqCmd::Process { data })?;
        Ok(())
    }

    pub fn check_pre_process(
        &self,
        timeout: Duration,
        timer: &TimerRef,
    ) -> Result<RtPreProcessResult> {
        let r = timer.with(|| self.handle.recv_with_timeout(timeout));
        match r {
            Ok(SeqResp::PreProcess {
                json,
                suspend,
                attn_masks,
            }) => Ok(RtPreProcessResult {
                json: serde_json::from_str(&json)?,
                suspend,
                attn_masks,
            }),
            Ok(r) => Err(anyhow!("unexpected response (pre_process) {r:?}")),
            Err(e) => Err(e.into()),
        }
    }

    pub fn check_process(&self, timeout: Duration) -> Result<JSON> {
        match self.handle.recv_with_timeout(timeout) {
            Ok(SeqResp::Process { json }) => Ok(serde_json::from_str(&json)?),
            Ok(r) => Err(anyhow!("unexpected response (process) {r:?}")),
            Err(e) => Err(e.into()),
        }
    }
}

impl GroupCtx {
    fn add_worker(&mut self, query: IpcReceiver<GroupCmd>, query_resp: IpcSender<GroupResp>) {
        let id = self.cb_set.add(query).unwrap();
        self.workers.insert(id, query_resp);
    }

    fn dispatch_storage_cmd(&mut self, cmd: StorageCmd) -> StorageResp {
        match cmd {
            StorageCmd::ReadVar { name } => match self.variables.get(&name).map(|x| x.clone()) {
                None => StorageResp::VariableMissing {},
                Some((version, value)) => StorageResp::ReadVar { value, version },
            },
            StorageCmd::WriteVar {
                name,
                value,
                when_version_is,
                op,
            } => {
                let curr = self.variables.get(&name).map(|x| x.clone());
                match curr {
                    Some((prev_version, prev_val)) => match when_version_is {
                        Some(v) if v != prev_version => StorageResp::ReadVar {
                            version: prev_version,
                            value: prev_val,
                        },
                        _ => {
                            let value = match op {
                                StorageOp::Append => {
                                    let mut v = prev_val.clone();
                                    v.extend(value);
                                    v
                                }
                                StorageOp::Set => value,
                            };
                            let version = prev_version + 1;
                            self.variables.insert(name, (version, value));
                            StorageResp::WriteVar { version }
                        }
                    },

                    None => match when_version_is {
                        None => {
                            self.variables.insert(name, (1, value));
                            StorageResp::WriteVar { version: 1 }
                        }
                        Some(_) => StorageResp::VariableMissing {},
                    },
                }
            }
        }
    }

    fn dispatch_cmd(&mut self, cmd: GroupCmd) -> GroupResp {
        match cmd {
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
            GroupCmd::StorageCmd { cmd } => GroupResp::StorageResp {
                resp: self.dispatch_storage_cmd(cmd),
            },
        }
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            for ent in self.cb_set.select().unwrap() {
                match ent {
                    ipc::IpcSelectionResult::MessageReceived(id, msg) => {
                        let resp = self.dispatch_cmd(msg.to().unwrap());
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
    limits: AiciLimits,
    fork_worker: ForkerHandle,
}

fn forker_dispatcher(
    cmdch: IpcReceiver<ForkerCmd>,
    cmd_resp: IpcSender<ForkerResp>,
    wasm_ctx: WasmContext,
    shm: Shm,
) -> ! {
    // TODO we should waitpid() children here, so they don't become zombies
    loop {
        let cmd = cmdch.recv().unwrap();
        let cmd_id = cmd.id;

        match fork_child().unwrap() {
            ForkResult::Parent { handle } => {
                let query_handle = handle;
                match fork_child().unwrap() {
                    ForkResult::Parent { handle } => {
                        cmd_resp.send(ForkerResp(handle)).unwrap();
                    }
                    ForkResult::Child { cmd, cmd_resp } => {
                        let pre_timer = wasm_ctx.timers.new_timer("pre_outer");
                        let mut w_ctx = SeqCtx {
                            id: cmd_id,
                            cmd,
                            cmd_resp,
                            wasm_ctx,
                            shm: Rc::new(shm),
                            query: Some(query_handle),
                            inst_id: 424242,
                            modinst: None,
                            pre_timer,
                        };
                        w_ctx.dispatch_loop()
                    }
                }
            }
            ForkResult::Child { cmd, cmd_resp } => {
                let mut grp_ctx = GroupCtx {
                    variables: HashMap::new(),
                    workers: HashMap::new(),
                    cb_set: IpcReceiverSet::new().unwrap(),
                };
                grp_ctx.add_worker(cmd, cmd_resp);
                grp_ctx.dispatch_loop()
            }
        }
    }
}

extern "C" fn clean_exit(_: libc::c_int) {
    std::process::exit(0);
}

pub fn stop_process() -> ! {
    let pgid = unsafe { libc::getpgrp() };
    unsafe { libc::kill(-pgid, libc::SIGUSR1) };
    std::thread::sleep(Duration::from_millis(500));
    panic!("didn't die");
}

pub fn bench_ipc() {
    let timers = TimerSet::new();
    match fork_child().unwrap() {
        ForkResult::Parent { handle } => {
            let cnt = 100;
            let timer = timers.new_timer("ipc");
            for idx in 0..cnt {
                let r = timer.with(|| handle.send_cmd(idx).unwrap());
                assert!(r == 2 * idx);
                std::thread::sleep(Duration::from_millis(5));
            }
            println!("ipc_channel {}", timers);
            handle.kill();
        }
        ForkResult::Child { cmd, cmd_resp } => loop {
            let r = cmd.recv().unwrap();
            cmd_resp.send(2 * r).unwrap();
        },
    }
}

impl WorkerForker {
    pub fn new(wasm_ctx: WasmContext, shm: Shm) -> Self {
        // create a new process group
        let pid = unsafe { libc::getpid() };
        unsafe {
            let r = libc::setpgid(pid, pid);
            assert!(r >= 0);
        };

        let limits = wasm_ctx.limits.clone();

        match fork_child().unwrap() {
            ForkResult::Parent { handle } => {
                unsafe { libc::signal(libc::SIGUSR1, clean_exit as usize) };
                WorkerForker {
                    fork_worker: handle,
                    limits,
                }
            }
            ForkResult::Child { cmd, cmd_resp } => forker_dispatcher(cmd, cmd_resp, wasm_ctx, shm),
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
        let res = SeqWorkerHandle {
            req_id: req.req_id.clone(),
            handle: resp.0,
        };
        res.handle.send_cmd_expect_ok(
            SeqCmd::Instantiate {
                module_path,
                module_id: req.module_id.clone(),
                module_arg,
                prompt_str,
                prompt_toks,
            },
            Duration::from_millis(self.limits.max_init_ms),
        )?;
        Ok(res)
    }
}
