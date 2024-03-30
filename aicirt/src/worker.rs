use crate::{
    api::ModuleInstId,
    bindings::MidProcessArg,
    hostimpl::AiciLimits,
    moduleinstance::{ModuleInstance, WasmContext},
    setup_bg_worker_pool,
    shm::Shm,
    InstantiateReq, UserError,
};
use aici_abi::{toktrie, StorageCmd, StorageOp, StorageResp};
use aicirt::{
    api::SequenceResult,
    bindings::*,
    futexshm::{TypedClient, TypedClientHandle, TypedServer},
    set_max_priority,
    shm::{ShmAllocator, Unlink},
    user_error,
    variables::Variables,
};
use anyhow::{anyhow, Result};
use libc::pid_t;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const QUICK_OP_MS: u64 = 3;
const QUICK_OP_RETRY_MS: u64 = 100;

#[derive(Serialize, Deserialize, Debug)]
pub enum GroupCmd {
    StorageCmd { cmd: StorageCmd },
}

#[derive(Serialize, Deserialize, Debug)]
pub enum GroupResp {
    StorageResp { resp: StorageResp },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WireProcessHandle<Cmd, Resp> {
    pid: pid_t,
    cmd: TypedClientHandle<Cmd, Resp>,
}

pub struct ProcessHandle<Cmd, Resp> {
    pid: pid_t,
    // should this be RefCell?
    cmd: Mutex<TypedClient<Cmd, Resp>>,
}

impl<Cmd, Resp> WireProcessHandle<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    pub fn to_client(self) -> ProcessHandle<Cmd, Resp> {
        ProcessHandle {
            pid: self.pid,
            cmd: Mutex::new(self.cmd.to_client()),
        }
    }
}

pub enum ForkResult<Cmd, Resp> {
    Parent {
        handle: WireProcessHandle<Cmd, Resp>,
    },
    Child {
        server: TypedServer<Cmd, Resp>,
    },
}

// This is visible with 'ps -o comm'
pub fn set_process_name(name: &str) {
    let name = std::ffi::CString::new(name).unwrap();
    let _ = name;
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_NAME, name.as_ptr() as usize, 0, 0, 0);
    }
}

pub fn fork_child<Cmd, Resp>(limits: &AiciLimits) -> Result<ForkResult<Cmd, Resp>>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    let shm_name = format!(
        "/aici-{}",
        uuid::Uuid::new_v4().to_string().replace('-', "")
    );
    // 31 is max length for shm name on macos
    let shm_name = shm_name[0..31].to_string();
    let shm = Shm::new(&shm_name, limits.ipc_shm_bytes, Unlink::Pre)?;

    let pid = unsafe { libc::fork() };

    if pid < 0 {
        return Err(anyhow!("fork failed"));
    }

    if pid == 0 {
        let server = TypedServer::<Cmd, Resp>::new(shm);
        Ok(ForkResult::Child { server })
    } else {
        // we drop shm, so it's unmapped
        drop(shm);
        // cmd.to_client() will unlink the shm - thus we need it constructed before
        // so there is no race between creation in the child and unlinking
        let cmd = TypedClientHandle::<Cmd, Resp>::new(shm_name, limits.ipc_shm_bytes);
        Ok(ForkResult::Parent {
            handle: WireProcessHandle { pid, cmd },
        })
    }
}

type ForkerHandle = ProcessHandle<ForkerCmd, ForkerResp>;
type SeqHandle = ProcessHandle<SeqCmd, SeqResp>;
type WireSeqHandle = WireProcessHandle<SeqCmd, SeqResp>;
pub type GroupHandle = ProcessHandle<GroupCmd, GroupResp>;

#[derive(Serialize, Deserialize, Debug)]
struct ForkerCmd {
    id: String,
    for_compile: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RtMidProcessArg {
    pub op: MidProcessArg,
}

#[derive(Serialize, Deserialize, Debug)]
struct ForkerResp(WireSeqHandle);

#[derive(Serialize, Deserialize, Debug)]
enum SeqCmd {
    GetCommsPid {},
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
    MidProcess {
        data: RtMidProcessArg,
    },
    RunMain {},
    Compile {
        wasm: Vec<u8>,
    },
}

impl SeqCmd {
    pub fn tag(&self) -> &'static str {
        match self {
            SeqCmd::GetCommsPid {} => "get_comms_pid",
            SeqCmd::Instantiate { .. } => "instantiate",
            SeqCmd::Fork { .. } => "fork",
            SeqCmd::SetId { .. } => "set_id",
            SeqCmd::MidProcess { .. } => "process",
            SeqCmd::RunMain {} => "run_main",
            SeqCmd::Compile { .. } => "compile",
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
enum SeqResp {
    Fork { handle: WireSeqHandle },
    CommsPid { pid: pid_t },
    Ok {},
    InitPrompt { json: String },
    PostPreProcess { post_json: String, pre_json: String },
    MidProcess { json: String },
    Compile { binary: Vec<u8> },
    Error { msg: String, is_user_error: bool },
}

pub enum Timeout {
    Quick,
    Strict(Duration),
    Speculative(Duration),
}

impl Timeout {
    pub fn from_millis(millis: u64) -> Self {
        Timeout::Strict(Duration::from_millis(millis))
    }
}

impl<Cmd, Resp> ProcessHandle<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize + Debug,
    Resp: for<'d> Deserialize<'d> + Serialize + Debug,
{
    pub fn just_send(&self, cmd: Cmd) -> Result<()> {
        log::trace!("send {cmd:?}");
        self.cmd.lock().unwrap().send_req(cmd)?;
        Ok(())
    }

    pub fn recv(&self) -> Result<Resp> {
        match self.cmd.lock().unwrap().recv_resp(Duration::MAX) {
            Some(r) => {
                log::trace!("recv {r:?}");
                Ok(r)
            }
            None => Err(anyhow!("timeout")),
        }
    }

    pub fn send_cmd(&self, cmd: Cmd) -> Result<Resp> {
        self.just_send(cmd)?;
        // otherwise we may get a response for someone else
        // this is only used for group commands and fork
        self.cmd.lock().unwrap().wait_for_reception();
        self.recv()
    }

    pub fn kill(&self) -> libc::c_int {
        assert!(self.pid != 0);
        unsafe { libc::kill(self.pid, libc::SIGKILL) }
    }

    fn recv_with_timeout(&self, lbl: &str, timeout: Timeout) -> Result<Resp> {
        let t0 = Instant::now();
        let d = match timeout {
            Timeout::Quick => Duration::from_millis(QUICK_OP_MS),
            Timeout::Strict(d) => d,
            Timeout::Speculative(d) => d,
        };
        let r = match timeout {
            Timeout::Quick => match self.recv_with_timeout_inner(d) {
                None => {
                    match self.recv_with_timeout_inner(Duration::from_millis(QUICK_OP_RETRY_MS)) {
                        Some(r) => {
                            let dur = t0.elapsed();
                            log::warn!("{lbl}: slow quick op: {dur:?}");
                            Some(r)
                        }
                        None => None,
                    }
                }
                r => r,
            },
            _ => self.recv_with_timeout_inner(d),
        };

        match r {
            Some(r) => Ok(r),
            None => match r {
                Some(r) => Ok(r),
                None => {
                    if !matches!(timeout, Timeout::Speculative(_)) {
                        let dur = t0.elapsed();
                        log::error!("{lbl}: timeout {dur:?} (allowed: {d:?})");
                    }
                    Err(anyhow!("timeout"))
                }
            },
        }
    }

    fn recv_with_timeout_inner(&self, timeout: Duration) -> Option<Resp> {
        match self.cmd.lock().unwrap().recv_resp(timeout) {
            Some(r) => {
                log::trace!("recv_with_timeout {r:?}");
                Some(r)
            }
            None => None,
        }
    }
}

impl SeqHandle {
    fn send_cmd_expect_ok(&self, cmd: SeqCmd, timeout: Timeout) -> Result<()> {
        let tag = cmd.tag();
        self.just_send(cmd)?;
        match self.seq_recv_with_timeout(tag, timeout) {
            Ok(SeqResp::Ok {}) => Ok(()),
            Ok(r) => Err(anyhow!("unexpected response (not OK) {r:?}")),
            Err(e) => Err(e.into()),
        }
    }

    fn seq_recv_with_timeout(&self, lbl: &str, timeout: Timeout) -> Result<SeqResp> {
        match self.recv_with_timeout(lbl, timeout) {
            Ok(SeqResp::Error { msg, is_user_error }) => {
                if is_user_error {
                    Err(user_error!("{lbl}: {msg}"))
                } else {
                    Err(anyhow!("{lbl}: {msg}"))
                }
            }
            r => r,
        }
    }

    fn send_cmd_with_timeout(&self, cmd: SeqCmd, timeout: Timeout) -> Result<SeqResp> {
        let tag = cmd.tag();
        self.just_send(cmd)?;
        self.seq_recv_with_timeout(tag, timeout)
    }
}

fn ok() -> Result<SeqResp> {
    Ok(SeqResp::Ok {})
}

impl SeqCtx {
    fn dispatch_one(&mut self, cmd: SeqCmd) -> Result<SeqResp> {
        match cmd {
            SeqCmd::GetCommsPid {} => Ok(SeqResp::CommsPid {
                pid: self.query.as_ref().unwrap().pid,
            }),
            SeqCmd::Compile { wasm } => {
                let inp_len = wasm.len();
                let start_time = Instant::now();
                let binary = self.wasm_ctx.engine.precompile_component(&wasm)?;
                log::info!(
                    "WASM compile done; {}k -> {}k; {:?}",
                    inp_len / 1024,
                    binary.len() / 1024,
                    Instant::now() - start_time
                );
                Ok(SeqResp::Compile { binary })
            }
            SeqCmd::Fork { inst_id } => {
                match fork_child(&self.wasm_ctx.limits)? {
                    ForkResult::Parent { handle } => Ok(SeqResp::Fork { handle }),
                    ForkResult::Child { server } => {
                        set_max_priority();
                        self.server = server;
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
                let component = self.wasm_ctx.deserialize_component(module_path)?;
                let _ = module_id;
                let ch = std::mem::take(&mut self.query);
                let mut inst = ModuleInstance::new(
                    424242,
                    self.wasm_ctx.clone(),
                    component,
                    module_arg,
                    ch.unwrap(),
                )?;
                let prompt_toks = if let Some(t) = prompt_toks {
                    t
                } else {
                    // TODO llama hack (doesn't apply in rllm)
                    let ps = &prompt_str.as_ref().unwrap();
                    let p = if ps.len() == 0 {
                        "<s>".to_string()
                    } else {
                        "<s> ".to_string() + ps
                    };
                    inst.tokenize(&p)?
                };
                self.modinst = Some(inst);
                let r = self.mutinst().setup(prompt_toks);
                Ok(SeqResp::InitPrompt {
                    json: serde_json::to_string(&r)?,
                })
            }
            SeqCmd::SetId { inst_id } => {
                self.inst_id = inst_id;
                self.mutinst().set_id(inst_id);
                ok()
            }
            SeqCmd::MidProcess { data } => {
                let res = self.mutinst().mid_process(data);
                Ok(SeqResp::MidProcess {
                    json: serde_json::to_string(&res)?,
                })
            }
            SeqCmd::RunMain {} => {
                // TODO
                // self.mutinst().run_main()?;
                ok()
            }
        }
    }

    fn mutinst(&mut self) -> &mut ModuleInstance {
        self.modinst.as_mut().unwrap()
    }

    // we may want to do this in future, but for now only group cmd is storage
    #[allow(dead_code)]
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
            let cmd = self
                .server
                .recv_req(self.wasm_ctx.limits.busy_wait_duration);
            log::trace!("seq recv {cmd:?}");
            let resp = match self.dispatch_one(cmd) {
                Ok(v) => v,
                Err(e) => SeqResp::Error {
                    msg: UserError::maybe_stacktrace(&e),
                    is_user_error: UserError::is_self(&e),
                },
            };
            self.server.send_resp(resp);
        }
    }
}

struct GroupCtx {
    variables: Variables,
    server: TypedServer<GroupCmd, GroupResp>,
    limits: AiciLimits,
}

struct SeqCtx {
    #[allow(dead_code)]
    id: String,
    server: TypedServer<SeqCmd, SeqResp>,
    wasm_ctx: WasmContext,
    query: Option<GroupHandle>,
    inst_id: ModuleInstId,
    modinst: Option<ModuleInstance>,
    shm: Rc<ShmAllocator>,
}

struct CommsPid {
    pid: pid_t,
}

impl Drop for CommsPid {
    fn drop(&mut self) {
        unsafe { libc::kill(self.pid, libc::SIGKILL) };
    }
}

pub struct SeqWorkerHandle {
    pub req_id: String,
    handle: SeqHandle,
    comms_pid: Option<Arc<CommsPid>>,
}

impl Drop for SeqWorkerHandle {
    fn drop(&mut self) {
        self.handle.kill();
    }
}

impl SeqWorkerHandle {
    pub fn set_id(&self, id: ModuleInstId) -> Result<()> {
        self.handle
            .send_cmd_expect_ok(SeqCmd::SetId { inst_id: id }, Timeout::Quick)
    }

    pub fn run_main(&self) -> Result<()> {
        self.handle
            .send_cmd_expect_ok(SeqCmd::RunMain {}, Timeout::from_millis(120_000))
    }

    pub fn fork(&self, target_id: ModuleInstId) -> Result<SeqWorkerHandle> {
        match self
            .handle
            .send_cmd_with_timeout(SeqCmd::Fork { inst_id: target_id }, Timeout::Quick)?
        {
            SeqResp::Fork { handle } => {
                let res = SeqWorkerHandle {
                    req_id: self.req_id.clone(),
                    handle: handle.to_client(),
                    comms_pid: self.comms_pid.clone(),
                };
                match res.handle.recv_with_timeout("r-fork", Timeout::Quick)? {
                    SeqResp::Ok {} => Ok(res),
                    r => Err(anyhow!("unexpected response (fork, child) {r:?}")),
                }
            }
            r => Err(anyhow!("unexpected response (fork) {r:?}")),
        }
    }

    pub fn start_process(&self, data: RtMidProcessArg) -> Result<()> {
        self.handle.just_send(SeqCmd::MidProcess { data })?;
        Ok(())
    }

    pub fn check_process(&self, timeout: Duration) -> Result<SequenceResult<MidProcessResult>> {
        match self
            .handle
            .seq_recv_with_timeout("r-process", Timeout::Speculative(timeout))
        {
            Ok(SeqResp::MidProcess { json }) => Ok(serde_json::from_str(&json)?),
            Ok(r) => Err(anyhow!("unexpected response (process) {r:?}")),
            Err(e) => Err(e.into()),
        }
    }
}

impl GroupCtx {
    fn dispatch_storage_cmd(&mut self, cmd: StorageCmd) -> StorageResp {
        self.variables.process_cmd(cmd)
    }

    fn dispatch_cmd(&mut self, cmd: GroupCmd) -> GroupResp {
        match cmd {
            GroupCmd::StorageCmd { cmd } => GroupResp::StorageResp {
                resp: self.dispatch_storage_cmd(cmd),
            },
        }
    }

    fn dispatch_loop(&mut self) -> ! {
        loop {
            let msg = self.server.recv_req(self.limits.busy_wait_duration);
            let resp = self.dispatch_cmd(msg);
            self.server.send_resp(resp);
        }
    }
}

pub struct WorkerForker {
    limits: AiciLimits,
    fork_worker: ForkerHandle,
}

fn forker_dispatcher(
    mut server: TypedServer<ForkerCmd, ForkerResp>,
    wasm_ctx: WasmContext,
    shm: Rc<ShmAllocator>,
) -> ! {
    set_process_name("aicirt-forker");
    loop {
        // wait for any children that might have exited to prevent zombies
        loop {
            let mut status = 0;
            let pid = unsafe { libc::waitpid(-1, &mut status, libc::WNOHANG) };
            if pid > 0 {
                if libc::WIFEXITED(status) {
                    let exit_code = libc::WEXITSTATUS(status);
                    log::debug!("Child {} exited with code {}", pid, exit_code);
                } else {
                    log::debug!("Child {} exited or sth", pid);
                }
            } else {
                // no (more) children; stop
                break;
            }
        }

        let cmd = server.recv_req(wasm_ctx.limits.busy_wait_duration);
        let cmd_id = cmd.id;
        let for_compile = cmd.for_compile;

        // fork the seq worker first
        match fork_child(&wasm_ctx.limits).unwrap() {
            ForkResult::Parent { handle } => {
                server.send_resp(ForkerResp(handle));
            }
            ForkResult::Child { server } => {
                let _pre_timer = wasm_ctx.timers.new_timer("pre_outer");
                let mut w_ctx = SeqCtx {
                    id: cmd_id,
                    server,
                    wasm_ctx,
                    shm,
                    query: None,
                    inst_id: 424242,
                    modinst: None,
                };

                if for_compile {
                    setup_bg_worker_pool();
                    w_ctx.dispatch_loop();
                }

                // and the seq worker then forks the communication process
                // this way we don't have to send query_handle to the seq worker
                // (inheriting it from fork doesn't work on macOS)
                match fork_child(&w_ctx.wasm_ctx.limits).unwrap() {
                    ForkResult::Parent { handle } => {
                        set_process_name("aicirt-seq");
                        set_max_priority();
                        w_ctx.query = Some(handle.to_client());
                        w_ctx.dispatch_loop()
                    }
                    ForkResult::Child { server } => {
                        set_process_name("aicirt-comms");
                        set_max_priority();
                        let mut grp_ctx = GroupCtx {
                            variables: Variables::default(),
                            server,
                            limits: w_ctx.wasm_ctx.limits,
                        };
                        grp_ctx.dispatch_loop()
                    }
                }
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

// pub fn bench_ipc(limits: &AiciLimits) {
//     let dur = Duration::from_millis(200);
//     let timers = TimerSet::new();
//     match fork_child(limits).unwrap() {
//         ForkResult::Parent { handle } => {
//             let cnt = 100;
//             let timer = timers.new_timer("ipc");
//             for idx in 0..cnt {
//                 let r = timer.with(|| handle.send_cmd(idx).unwrap());
//                 assert!(r == 2 * idx);
//                 std::thread::sleep(Duration::from_millis(5));
//             }
//             println!("ipc_channel {}", timers);
//             handle.kill();
//         }
//         ForkResult::Child { cmd, cmd_resp } => loop {
//             let r = busy_recv(&cmd, &dur).unwrap();
//             // let r = cmd.recv().unwrap();
//             cmd_resp.send(2 * r).unwrap();
//         },
//     }
// }

impl WorkerForker {
    pub fn new(wasm_ctx: WasmContext, shm: Rc<ShmAllocator>) -> Self {
        // create a new process group
        let pid = unsafe { libc::getpid() };
        unsafe {
            let r = libc::setpgid(pid, pid);
            assert!(r >= 0);
        };

        let limits = wasm_ctx.limits.clone();

        match fork_child(&limits).unwrap() {
            ForkResult::Parent { handle } => {
                unsafe { libc::signal(libc::SIGUSR1, clean_exit as usize) };
                WorkerForker {
                    fork_worker: handle.to_client(),
                    limits,
                }
            }
            ForkResult::Child { server } => forker_dispatcher(server, wasm_ctx, shm),
        }
    }

    pub fn instantiate(
        &self,
        req: InstantiateReq,
        module_path: PathBuf,
    ) -> Result<(SeqWorkerHandle, SequenceResult<InitPromptResult>)> {
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
                        .ok_or_else(|| anyhow!("expecting string or int array as prompt"))?
                        .iter()
                        .map(|x| -> Result<u32> {
                            x.as_u64()
                                .ok_or_else(|| anyhow!("expecting number as token"))?
                                .try_into()
                                .map_err(|e: std::num::TryFromIntError| anyhow!(e))
                        })
                        .collect::<Result<Vec<u32>>>()?,
                ),
            )
        };

        let resp = self.fork_worker.send_cmd(ForkerCmd {
            id: req.req_id.clone(),
            for_compile: false,
        })?;
        let mut res = SeqWorkerHandle {
            req_id: req.req_id.clone(),
            handle: resp.0.to_client(),
            comms_pid: None,
        };
        let comms_pid = match res
            .handle
            .send_cmd_with_timeout(SeqCmd::GetCommsPid {}, Timeout::Quick)?
        {
            SeqResp::CommsPid { pid } => pid,
            r => return Err(anyhow!("unexpected response (get comms pid) {r:?}")),
        };
        res.comms_pid = Some(Arc::new(CommsPid { pid: comms_pid }));
        match res.handle.send_cmd_with_timeout(
            SeqCmd::Instantiate {
                module_path,
                module_id: req.module_id.clone(),
                module_arg,
                prompt_str,
                prompt_toks,
            },
            Timeout::from_millis(self.limits.max_init_ms),
        )? {
            SeqResp::InitPrompt { json } => {
                let r: SequenceResult<InitPromptResult> = serde_json::from_str(&json)?;
                Ok((res, r))
            }
            r => Err(anyhow!("unexpected response (init prompt) {r:?}")),
        }
    }

    pub fn compile(&self, wasm: Vec<u8>) -> Result<Vec<u8>> {
        let id = "compile".to_string();
        let resp = self.fork_worker.send_cmd(ForkerCmd {
            id: id.clone(),
            for_compile: true,
        })?;

        // res.drop() kills handle
        let res = SeqWorkerHandle {
            req_id: id.clone(),
            handle: resp.0.to_client(),
            comms_pid: None,
        };
        match res.handle.send_cmd_with_timeout(
            SeqCmd::Compile { wasm },
            Timeout::from_millis(self.limits.max_compile_ms),
        )? {
            SeqResp::Compile { binary } => Ok(binary),
            r => Err(anyhow!("unexpected response (compile) {r:?}")),
        }
    }
}
