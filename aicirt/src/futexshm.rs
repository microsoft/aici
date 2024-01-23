use crate::shm::{Shm, Unlink};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    ptr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

#[cfg(target_os = "linux")]
use linux_futex::AsFutex;
#[cfg(target_os = "linux")]
type Futex = linux_futex::Futex<linux_futex::Shared>;

#[cfg(target_os = "macos")]
use crate::macos::AsFutex;
#[cfg(target_os = "macos")]
type Futex = crate::macos::Futex;

fn futex_at(shm: &Shm, off: usize) -> &'static Futex {
    assert!(shm.size >= off + 4);
    unsafe { AtomicU32::from_ptr(shm.ptr_at(off) as *mut u32).as_futex() }
}

pub struct WrMsgCounter {
    futex: &'static Futex,
    #[allow(dead_code)]
    shm: Shm,
}

impl WrMsgCounter {
    pub fn new(shm: Shm) -> Self {
        Self {
            futex: futex_at(&shm, 0),
            shm,
        }
    }

    pub fn get(&self) -> u32 {
        self.futex.value.load(Ordering::Acquire)
    }

    pub fn inc(&self) {
        let _ = self.futex.value.fetch_add(1, Ordering::AcqRel);
        let _ = self.futex.wake(i32::MAX);
    }
}

pub struct RdMsgCounter {
    futex: &'static Futex,
    #[allow(dead_code)]
    shm: Arc<Shm>,
}

impl RdMsgCounter {
    pub fn new(shm: Arc<Shm>) -> Self {
        Self {
            futex: futex_at(&shm, 0),
            shm,
        }
    }

    pub fn read(&self) -> u32 {
        self.futex.value.load(Ordering::Acquire)
    }

    pub fn wait(&self, val: u32) {
        let _ = self.futex.wait(val);
    }
}

struct Channel {
    wr_len: &'static Futex,
    rd_len: &'static Futex,
    shm: Shm,
}

impl Channel {
    fn new(shm: Shm, swap: bool) -> Self {
        assert!(shm.size > 4096 * 2);
        let len0 = futex_at(&shm, 0);
        let len1 = futex_at(&shm, shm.size / 2);
        let (wr_len, rd_len) = if swap { (len1, len0) } else { (len0, len1) };
        Self {
            wr_len,
            rd_len,
            shm,
        }
    }

    pub fn max_msg_size(&self) -> usize {
        self.shm.size / 2 - 16
    }

    pub fn fits_msg(&self, msg: &[u8]) -> Result<()> {
        if msg.len() > self.max_msg_size() {
            return Err(anyhow!(
                "msg too large; {} > {}",
                msg.len(),
                self.max_msg_size()
            ));
        }
        Ok(())
    }

    pub fn wait_for_reception(&mut self) {
        let val = &self.wr_len.value;
        while val.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
    }

    pub fn write_msg(&mut self, msg: &[u8]) -> Result<()> {
        self.fits_msg(msg)?;

        let val = &self.wr_len.value;

        loop {
            while val.load(Ordering::Acquire) != 0 {
                std::hint::spin_loop();
            }
            if val
                .compare_exchange_weak(0, u32::MAX, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }

        unsafe {
            ptr::copy_nonoverlapping(
                msg.as_ptr(),
                self.wr_len.value.as_ptr().add(1) as *mut u8,
                msg.len(),
            );
        }

        val.store(msg.len() as u32, Ordering::Release);
        let _n = self.wr_len.wake(i32::MAX);
        // if n > 0 {
        //     log::warn!("wake up {} threads", n);
        // }

        Ok(())
    }

    pub fn read_len(&self) -> usize {
        let r = self.rd_len.value.load(Ordering::Acquire);
        if r == u32::MAX {
            0
        } else {
            r as usize
        }
    }

    pub fn wait_for_len(
        &self,
        mut spin_duration: Duration,
        futex_duration: Option<Duration>,
    ) -> Option<usize> {
        let mut len = self.read_len();
        if len != 0 {
            return Some(len);
        }
        spin_duration = std::cmp::min(spin_duration, Duration::from_secs(365 * 24 * 3600));
        let deadline = Instant::now() + spin_duration;
        while Instant::now() < deadline {
            len = self.read_len();
            if len != 0 {
                break;
            }
            std::hint::spin_loop();
        }
        if len != 0 {
            return Some(len);
        }
        if let Some(futex_duration) = futex_duration {
            let v = self.rd_len.value.load(Ordering::Acquire);
            if v == 0 || v == u32::MAX {
                if futex_duration == Duration::MAX {
                    let _ = self.rd_len.wait(v);
                } else {
                    let _ = self.rd_len.wait_for(v, futex_duration);
                }
            }
            len = self.read_len();
        }
        if len != 0 {
            Some(len)
        } else {
            None
        }
    }

    pub fn read_msg(
        &mut self,
        spin_duration: Duration,
        futex_duration: Option<Duration>,
    ) -> Option<Vec<u8>> {
        let msg_len = self.wait_for_len(spin_duration, futex_duration)?;

        if msg_len > self.max_msg_size() {
            panic!("read: shm too small {} < {}", msg_len, self.max_msg_size());
        }

        let mut res = vec![0u8; msg_len];
        unsafe {
            ptr::copy_nonoverlapping(
                self.rd_len.value.as_ptr().add(1) as *const u8,
                res.as_mut_ptr(),
                msg_len,
            )
        };
        self.rd_len.value.store(0, Ordering::Release);

        Some(res)
    }
}

pub struct ClientChannel {
    channel: Channel,
}

impl ClientChannel {
    pub fn new(shm: Shm) -> Self {
        Self {
            channel: Channel::new(shm, false),
        }
    }

    pub fn send_req(&mut self, msg: &[u8]) -> Result<()> {
        self.channel.write_msg(msg)
    }

    pub fn recv_resp(&mut self, timeout: Duration) -> Option<Vec<u8>> {
        self.channel.read_msg(timeout, None)
    }

    pub fn recv_resp2(
        &mut self,
        busy_timeout: Duration,
        futex_timeout: Duration,
    ) -> Option<Vec<u8>> {
        self.channel.read_msg(busy_timeout, Some(futex_timeout))
    }
}

pub struct ServerChannel {
    channel: Channel,
    #[allow(dead_code)]
    msg_cnt: RdMsgCounter,
}

impl ServerChannel {
    pub fn new(shm: Shm, cnt_shm: Arc<Shm>) -> Self {
        Self {
            channel: Channel::new(shm, true),
            msg_cnt: RdMsgCounter::new(cnt_shm),
        }
    }

    pub fn recv_req(&mut self, busy_spin: Duration) -> Vec<u8> {
        self.channel
            .read_msg(busy_spin, Some(Duration::MAX))
            .unwrap()
        // if self.channel.wait_for_len(busy_spin, None).is_none() {
        //     loop {
        //         let val = self.msg_cnt.read();
        //         let len = self.channel.read_len();
        //         if len == 0 {
        //             self.msg_cnt.wait(val);
        //         } else {
        //             break;
        //         }
        //     }
        // }
        // self.channel.read_msg(busy_spin, None).unwrap()
    }

    pub fn send_resp(&mut self, msg: &[u8]) -> Result<()> {
        self.channel.write_msg(msg)
    }
}

pub struct TypedServer<Cmd, Resp> {
    channel: ServerChannel,
    _cmd: std::marker::PhantomData<Cmd>,
    _resp: std::marker::PhantomData<Resp>,
}

impl<Cmd, Resp> TypedServer<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    pub fn new(shm: Shm, cnt_shm: Arc<Shm>) -> Self {
        Self {
            channel: ServerChannel::new(shm, cnt_shm),
            _cmd: std::marker::PhantomData,
            _resp: std::marker::PhantomData,
        }
    }

    pub fn recv_req(&mut self, busy_spin: Duration) -> Cmd {
        let msg = self.channel.recv_req(busy_spin);
        bincode::deserialize(&msg).unwrap()
    }

    pub fn send_resp(&mut self, resp: Resp) {
        let msg = bincode::serialize(&resp).unwrap();
        self.channel.send_resp(&msg).unwrap();
    }
}

pub struct TypedClient<Cmd, Resp> {
    channel: ClientChannel,
    _cmd: std::marker::PhantomData<Cmd>,
    _resp: std::marker::PhantomData<Resp>,
}

impl<Cmd, Resp> TypedClient<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    pub fn new(shm: Shm) -> Self {
        Self {
            channel: ClientChannel::new(shm),
            _cmd: std::marker::PhantomData,
            _resp: std::marker::PhantomData,
        }
    }

    pub fn wait_for_reception(&mut self) {
        self.channel.channel.wait_for_reception();
    }

    pub fn send_req(&mut self, cmd: Cmd) -> Result<()> {
        let msg = bincode::serialize(&cmd).unwrap();
        self.channel.send_req(&msg)
    }

    pub fn recv_resp(&mut self, timeout: Duration) -> Option<Resp> {
        let msg = self.channel.recv_resp(timeout)?;
        Some(bincode::deserialize(&msg).unwrap())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypedClientHandle<Cmd, Resp> {
    shm_name: String,
    shm_size: usize,
    _cmd: std::marker::PhantomData<Cmd>,
    _resp: std::marker::PhantomData<Resp>,
}

impl<Cmd, Resp> TypedClientHandle<Cmd, Resp>
where
    Cmd: for<'d> Deserialize<'d> + Serialize,
    Resp: for<'d> Deserialize<'d> + Serialize,
{
    pub fn new(shm_name: String, shm_size: usize) -> Self {
        Self {
            shm_name,
            shm_size,
            _cmd: std::marker::PhantomData,
            _resp: std::marker::PhantomData,
        }
    }

    pub fn to_client(self) -> TypedClient<Cmd, Resp> {
        let shm = Shm::new(&self.shm_name, self.shm_size, Unlink::Post).unwrap();
        TypedClient::new(shm)
    }
}
