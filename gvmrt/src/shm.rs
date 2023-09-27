use std::ffi::CString;
use std::io;
use std::ptr;

use anyhow::{anyhow, ensure, Result};
use log::info;

pub struct Shm {
    addr: *mut u8,
    size: usize,
}

impl Shm {
    pub fn new(name: &str, size: usize) -> Result<Self> {
        ensure!(size > 1024);

        info!("shm_open: {} size={}k", name, size / 1024);

        let shm_name = CString::new(name).unwrap();
        let fd = unsafe { libc::shm_open(shm_name.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o666) };
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }

        unsafe {
            // ignore error
            let _ = libc::ftruncate(fd, size.try_into().unwrap());
        };

        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        let err = io::Error::last_os_error();
        unsafe { libc::close(fd) };

        if addr == libc::MAP_FAILED {
            return Err(err.into());
        }

        Ok(Self {
            addr: addr as *mut u8,
            size,
        })
    }

    pub fn split(&self, slice_size: usize) -> Result<Vec<&'static mut [u8]>> {
        let num = self.size / slice_size;
        ensure!(num > 0);
        Ok((0..self.size / slice_size)
            .map(|idx| unsafe {
                std::slice::from_raw_parts_mut(self.addr.add(idx * slice_size), slice_size)
            })
            .collect::<Vec<_>>())
    }

    pub fn fits_msg(&self, msg: &[u8]) -> Result<()> {
        if msg.len() + 4 > self.size {
            return Err(anyhow!("msg too large; {} + 4 > {}", msg.len(), self.size));
        }
        Ok(())
    }

    pub fn write_msg(&self, msg: &[u8]) -> Result<()> {
        self.fits_msg(msg)?;

        let msg_len = msg.len() as u32;
        let len_bytes = msg_len.to_le_bytes();

        unsafe {
            ptr::copy_nonoverlapping(len_bytes.as_ptr(), self.addr as *mut u8, 4);
            ptr::copy_nonoverlapping(msg.as_ptr(), (self.addr as *mut u8).add(4), msg.len());
        }

        Ok(())
    }

    pub fn read_msg(&self) -> Result<Vec<u8>> {
        let mut len_bytes = [0u8; 4];
        unsafe {
            ptr::copy_nonoverlapping(self.addr as *const u8, len_bytes.as_mut_ptr(), 4);
        }

        let msg_len = u32::from_le_bytes(len_bytes) as usize;

        if msg_len > self.size - 4 {
            return Err(anyhow!(
                "read: shm too small {} + 4 < {}",
                msg_len,
                self.size
            ));
        }

        let mut res = vec![0u8; msg_len];
        unsafe {
            ptr::copy_nonoverlapping((self.addr as *const u8).add(4), res.as_mut_ptr(), msg_len);
        };

        Ok(res)
    }
}

impl Drop for Shm {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.size);
        }
    }
}
