use anyhow::{anyhow, ensure, Result};
use std::{ffi::CString, io, ptr};

pub struct Shm {
    addr: *mut u8,
    pub size: usize,
}

unsafe impl Send for Shm {}
unsafe impl Sync for Shm {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unlink {
    None,
    Pre,
    Post,
}

impl Shm {
    pub fn anon(size: usize) -> Result<Self> {
        ensure!(size > 1024);
        log::trace!("shm_open anon: size={}k", size / 1024);
        Self::from_fd(-1, size)
    }

    fn from_fd(fd: i32, size: usize) -> Result<Self> {
        let mut flag = libc::MAP_SHARED;
        if fd == -1 {
            flag |= libc::MAP_ANONYMOUS;
        }
        let addr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_WRITE | libc::PROT_READ,
                flag,
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

    pub fn new(name: &str, size: usize, unlink: Unlink) -> Result<Self> {
        ensure!(size > 1024);

        log::trace!("shm_open: {} size={}k", name, size / 1024);

        let shm_name = CString::new(name).unwrap();
        if unlink == Unlink::Pre {
            unsafe { libc::shm_unlink(shm_name.as_ptr()) };
        }
        let fd = unsafe { libc::shm_open(shm_name.as_ptr(), libc::O_RDWR | libc::O_CREAT, 0o666) };
        if unlink == Unlink::Post {
            unsafe { libc::shm_unlink(shm_name.as_ptr()) };
        }
        if fd < 0 {
            return Err(io::Error::last_os_error().into());
        }

        unsafe {
            // ignore error
            let _ = libc::ftruncate(fd, size.try_into().unwrap());
        };

        Self::from_fd(fd, size)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn slice_at_byte_offset<T>(&self, off: usize, num_elts: usize) -> &'static mut [T] {
        let ptr = self.ptr_at(off);
        assert!(off + num_elts * std::mem::size_of::<T>() <= self.size);
        assert!(off % std::mem::align_of::<T>() == 0);
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, num_elts) }
    }

    pub fn ptr_at(&self, off: usize) -> *mut u8 {
        unsafe { self.addr.add(off) }
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

        unsafe {
            ptr::copy_nonoverlapping(msg.as_ptr(), (self.addr as *mut u8).add(4), msg.len());
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
            return Err(anyhow!(
                "read: shm too small {} + 4 < {}",
                msg_len,
                self.size
            ));
        }

        let mut res = vec![0u8; msg_len];
        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            ptr::copy_nonoverlapping((self.addr as *const u8).add(4), res.as_mut_ptr(), msg_len);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            ptr::write_volatile(self.addr as *mut u32, 0);
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
