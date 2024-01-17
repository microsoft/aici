use crate::shm::Shm;
use anyhow::{anyhow, Result};
use linux_futex::AsFutex;
use std::{
    ptr,
    sync::atomic::{AtomicU32, Ordering},
    time::{Duration, Instant},
};

type Futex = linux_futex::Futex<linux_futex::Shared>;

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
    shm: Shm,
}

impl RdMsgCounter {
    pub fn new(shm: Shm) -> Self {
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

const MSG_OFF: usize = 8;

impl Channel {
    fn new(shm: Shm, swap: bool) -> Self {
        assert!(shm.size > MSG_OFF);
        let len0 = futex_at(&shm, 0);
        let len1 = futex_at(&shm, 4);
        let (wr_len, rd_len) = if swap { (len1, len0) } else { (len0, len1) };
        Self {
            wr_len,
            rd_len,
            shm,
        }
    }

    pub fn fits_msg(&self, msg: &[u8]) -> Result<()> {
        if msg.len() + MSG_OFF > self.shm.size {
            return Err(anyhow!(
                "msg too large; {} + {MSG_OFF} > {}",
                msg.len(),
                self.shm.size
            ));
        }
        Ok(())
    }

    pub fn write_msg(&self, msg: &[u8]) -> Result<()> {
        self.fits_msg(msg)?;

        let msg_len = msg.len() as u32;

        unsafe {
            ptr::copy_nonoverlapping(msg.as_ptr(), self.shm.ptr_at(MSG_OFF), msg.len());
        }

        while self.wr_len.value.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }

        self.wr_len.value.store(msg_len, Ordering::Release);

        Ok(())
    }

    pub fn read_len(&self) -> usize {
        return self.rd_len.value.load(Ordering::Acquire) as usize;
    }

    pub fn wait_for_len(
        &self,
        spin_duration: Duration,
        futex_duration: Option<Duration>,
    ) -> Option<usize> {
        let mut len = self.read_len();
        if len != 0 {
            return Some(len);
        }
        let deadline = Instant::now() + spin_duration;
        while Instant::now() < deadline {
            len = self.read_len();
            if len != 0 {
                return Some(len);
            }
            std::hint::spin_loop();
        }
        if let Some(futex_duration) = futex_duration {
            if futex_duration == Duration::MAX {
                let _ = self.rd_len.wait(len as u32);
            } else {
                let _ = self.rd_len.wait_for(len as u32, futex_duration);
            };
            len = self.read_len();
        }
        if len != 0 {
            Some(len)
        } else {
            None
        }
    }

    pub fn read_msg(
        &self,
        spin_duration: Duration,
        futex_duration: Option<Duration>,
    ) -> Option<Vec<u8>> {
        let msg_len = self.wait_for_len(spin_duration, futex_duration)?;

        if msg_len > self.shm.size - MSG_OFF {
            panic!("read: shm too small {} + 4 < {}", msg_len, self.shm.size);
        }

        let mut res = vec![0u8; msg_len];
        unsafe { ptr::copy_nonoverlapping(self.shm.ptr_at(MSG_OFF), res.as_mut_ptr(), msg_len) };
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

    pub fn send_req(&self, msg: &[u8]) -> Result<()> {
        self.channel.write_msg(msg)
    }

    pub fn recv_resp(&self, timeout: Duration) -> Option<Vec<u8>> {
        self.channel.read_msg(timeout, None)
    }
}

pub struct ServerChannel {
    channel: Channel,
    msg_cnt: RdMsgCounter,
}

impl ServerChannel {
    pub fn new(shm: Shm, cnt_shm: Shm) -> Self {
        Self {
            channel: Channel::new(shm, true),
            msg_cnt: RdMsgCounter::new(cnt_shm),
        }
    }

    pub fn recv_req(&self, busy_spin: Duration) -> Vec<u8> {
        if self.channel.wait_for_len(busy_spin, None).is_none() {
            loop {
                let val = self.msg_cnt.read();
                let len = self.channel.read_len();
                if len == 0 {
                    self.msg_cnt.wait(val);
                } else {
                    break;
                }
            }
        }
        self.channel.read_msg(busy_spin, None).unwrap()
    }

    pub fn send_resp(&self, msg: &[u8]) -> Result<()> {
        self.channel.write_msg(msg)
    }
}
