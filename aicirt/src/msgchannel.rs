use std::time::Duration;

use crate::semaphore::Semaphore;
use crate::shm::Shm;
use anyhow::Result;

pub struct MessageChannel {
    shm: Shm,
    write_sem: Semaphore,
    read_sem: Semaphore,
}

unsafe impl Send for MessageChannel {}

impl MessageChannel {
    pub fn shm_name(name: &str) -> String {
        format!("{0}-shm", name)
    }

    pub fn new_cmd(name: &str, size: usize) -> Result<Self> {
        Self::new_ex(name, size, true)
    }

    pub fn new(name: &str, size: usize) -> Result<Self> {
        Self::new_ex(name, size, false)
    }

    fn new_ex(name: &str, size: usize, unlink: bool) -> Result<Self> {
        log::debug!("msgch: {} size={}k", name, size / 1024);

        let shm = Shm::new(&Self::shm_name(name), size, unlink)?;
        let write_sem = Semaphore::new(&format!("{0}-wr", name), 1, unlink)?;
        let read_sem = Semaphore::new(&format!("{0}-rd", name), 0, unlink)?;

        Ok(Self {
            shm,
            write_sem,
            read_sem,
        })
    }

    pub fn send(&self, msg: &[u8]) -> Result<()> {
        self.shm.fits_msg(msg)?;
        self.write_sem.wait()?;
        self.shm.write_msg(msg).unwrap();
        self.read_sem.post()?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn busy_reset(&self) {
        unsafe { std::ptr::write_volatile(self.shm.ptr_at(0) as *mut u32, 0) };
    }

    #[allow(dead_code)]
    pub fn busy_send(&self, msg: &[u8]) -> Result<()> {
        self.shm.fits_msg(msg)?;
        loop {
            let len = self.shm.read_len()?;
            if len != 0 {
                std::hint::spin_loop();
                continue;
            }
            return Ok(self.shm.write_msg(msg).unwrap());
        }
    }

    #[allow(dead_code)]
    pub fn busy_recv(&self) -> Result<Vec<u8>> {
        loop {
            let len = self.shm.read_len()?;
            if len == 0 {
                std::hint::spin_loop();
                continue;
            }
            let res = self.shm.read_msg();
            return res;
        }
    }

    pub fn recv(&self, busy_wait_duration: &Duration) -> Result<Vec<u8>> {
        self.read_sem.busy_wait(busy_wait_duration)?;
        let res = self.shm.read_msg();
        self.write_sem.post()?;
        res
    }
}
