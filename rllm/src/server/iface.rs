use aici_abi::bytes::limit_bytes;
use aicirt::{msgchannel::MessageChannel, shm::Shm};
use anyhow::Result;
use futures::future::select_all;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    process::{Child, Command},
    time::Duration,
};
use tokio::{signal::unix::SignalKind, task::JoinHandle};

use crate::Args;

pub struct CmdChannel {
    suff: String,
    cmd_pending: bool,
    cmd_ch: MessageChannel,
    resp_ch: MessageChannel,
    busy_wait_duration: Duration,
}

const M: usize = 1 << 20;

impl CmdChannel {
    pub fn new(
        json_size: usize,
        pref: &str,
        suff: &str,
        busy_wait_duration: Duration,
    ) -> Result<Self> {
        Ok(Self {
            suff: suff.to_string(),
            cmd_pending: false,
            cmd_ch: MessageChannel::new(&format!("{}cmd{}", pref, suff), json_size * M)?,
            resp_ch: MessageChannel::new(&format!("{}resp{}", pref, suff), json_size * M)?,
            busy_wait_duration,
        })
    }

    pub fn send_bytes(&mut self, data: &[u8]) -> Result<()> {
        assert!(!self.cmd_pending);
        self.cmd_pending = true;
        self.cmd_ch.send(data)?;
        Ok(())
    }

    pub fn exec<T: Serialize, R>(&mut self, op: &str, data: Option<T>) -> Result<()>
    where
        R: for<'d> Deserialize<'d>,
    {
        self.send(op, data)?;
        self.expect(&format!("cmd:{}", op))
    }

    pub fn send<T: Serialize>(&mut self, op: &str, data: Option<T>) -> Result<()> {
        let mut value = match data {
            Some(d) => serde_json::to_value(d)?,
            None => json!({}),
        };
        value["op"] = json!(op);
        let bytes = serde_json::to_vec(&value)?;
        self.send_bytes(&bytes)
    }

    pub fn expect<R>(&mut self, ctx: &str) -> Result<R>
    where
        R: for<'d> Deserialize<'d>,
    {
        assert!(self.cmd_pending);
        let bytes = self.resp_ch.recv(&self.busy_wait_duration)?;
        self.cmd_pending = false;
        let mut resp: Value = serde_json::from_slice(&bytes)?;
        if resp["type"] != "ok" {
            return Err(anyhow::anyhow!(
                "Bad response ({ctx}): {}",
                limit_bytes(&bytes, 500)
            ));
        }
        let data = resp
            .as_object_mut()
            .unwrap()
            .remove("data")
            .ok_or(anyhow::anyhow!(
                "Bad response ({ctx}) - no 'data': {}",
                limit_bytes(&bytes, 500)
            ))?;
        let resp = serde_json::from_value(data)?;
        Ok(resp)
    }
}

pub struct AiciRtIface {
    cmd: CmdChannel,
    side_cmd: CmdChannel,
    bin_shm: Shm,
    child: Child,
}

impl AiciRtIface {
    pub fn start_aicirt(args: &Args) -> Result<Self> {
        let busy_wait_time = Duration::from_millis(args.busy_wait_time);
        let shm_name = MessageChannel::shm_name(&args.shm_prefix) + "-bin";
        let cmd = CmdChannel::new(args.json_size, &args.shm_prefix, "", busy_wait_time)?;
        let side_cmd = CmdChannel::new(args.json_size, &args.shm_prefix, "-side", busy_wait_time)?;
        let bin_shm = Shm::new(&shm_name, args.bin_size * M)?;

        let child = Command::new(&args.aicirt)
            .arg("--tokenizer")
            .arg(&args.tokenizer)
            .arg("--json-size")
            .arg(&args.json_size.to_string())
            .arg("--bin-size")
            .arg(&args.bin_size.to_string())
            .arg("--name")
            .arg(&args.shm_prefix)
            .arg("--server")
            .spawn()?;

        let pid = child.id() as libc::c_int;
        let default_panic_hook = std::panic::take_hook();

        std::panic::set_hook(Box::new(move |panic_info| {
            eprintln!("killing {pid}");
            unsafe {
                libc::kill(-pid, libc::SIGTERM);
            }
            default_panic_hook(panic_info);
            std::process::exit(100);
        }));

        let _killer = tokio::spawn(async move {
            let sigs = vec![
                SignalKind::interrupt(),
                SignalKind::quit(),
                SignalKind::terminate(),
            ];

            let mut sigs = sigs
                .iter()
                .map(|s| tokio::signal::unix::signal(*s).unwrap())
                .collect::<Vec<_>>();

            loop {
                let futures: Vec<_> = sigs.iter_mut().map(|s| s.recv()).collect();
                let pinned_futures: Vec<_> = futures.into_iter().map(|f| Box::pin(f)).collect();
                select_all(pinned_futures).await;
                log::info!("Killing child process");
                unsafe {
                    libc::kill(-pid, libc::SIGTERM);
                }
            }
        });

        Ok(Self {
            cmd,
            side_cmd,
            bin_shm,
            child,
        })
    }
}
