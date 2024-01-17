use aici_abi::svob::SimpleVob;
use anyhow::{anyhow, ensure, Result};
use linux_futex::AsFutex;
use std::{io, ptr, sync::atomic::AtomicU32};

type Futex = linux_futex::Futex<linux_futex::Shared>;

pub struct FutexShm {
    addr: *mut u8,
    common_offset: usize,
    element_size: usize,
    used: SimpleVob,
    free: Vec<usize>,
    size: usize,
}

pub struct FutexChannel {
    futex: &'static Futex,
    size: usize,
    idx: usize,
    addr: *mut u8,
}

unsafe impl Send for FutexShm {}

const PAGE_SIZE: usize = 4096;

impl FutexShm {
    pub fn new(num_elts: usize, element_size: usize) -> Result<Self> {
        ensure!(num_elts > 0);
        ensure!(element_size > 0);
        ensure!(element_size % PAGE_SIZE == 0);

        let common_offset = PAGE_SIZE;
        let size = common_offset + num_elts * element_size;

        log::trace!("futex_open: size={}k", size / 1024);

        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };

        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }

        Ok(Self {
            addr: addr as *mut u8,
            element_size,
            common_offset,
            used: SimpleVob::alloc(num_elts),
            free: (0..num_elts).collect(),
            size,
        })
    }

    pub fn alloc(&mut self) -> Result<FutexChannel> {
        let idx = self.free.pop().ok_or_else(|| anyhow!("no free channels"))?;
        assert!(!self.used[idx]);
        self.used.set(idx as u32, true);
        let addr = unsafe { self.addr.add(self.common_offset + idx * self.element_size) };
        let futex = unsafe { AtomicU32::from_ptr(self.addr as *mut u32).as_futex() };
        Ok(FutexChannel {
            futex,
            addr,
            idx,
            size: self.element_size,
        })
    }

    pub fn free(&mut self, channel: FutexChannel) {
        let idx = channel.idx;
        assert!(self.used[idx]);
        self.used.set(idx as u32, false);
        self.free.push(idx);
    }
}

impl Drop for FutexShm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.size);
        }
    }
}

impl FutexChannel {
    pub fn fits_msg(&self, msg: &[u8]) -> Result<()> {
        if msg.len() + 4 > self.size {
            return Err(anyhow!("msg too large; {} + 4 > {}", msg.len(), self.size));
        }
        Ok(())
    }

    pub fn write_msg(&self, msg: &[u8]) -> Result<()> {
        self.fits_msg(msg)?;

        let msg_len = msg.len() as u32;

        unsafe {
            ptr::copy_nonoverlapping(msg.as_ptr(), (self.addr as *mut u8).add(4), msg.len());
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            ptr::write_volatile(self.addr as *mut u32, msg_len);
        }

        Ok(())
    }

    pub fn read_len(&self) -> Result<usize> {
        Ok(unsafe { ptr::read_volatile(self.addr as *const u32) } as usize)
    }

    pub fn read_msg(&self) -> Result<Vec<u8>> {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let msg_len = self.read_len()?;

        if msg_len > self.size - 4 {
            anyhow::bail!("read: shm too small {} + 4 < {}", msg_len, self.size);
        }

        let mut res = vec![0u8; msg_len];
        unsafe {
            ptr::copy_nonoverlapping((self.addr as *const u8).add(4), res.as_mut_ptr(), msg_len);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            ptr::write_volatile(self.addr as *mut u32, 0);
        };

        Ok(res)
    }
}
