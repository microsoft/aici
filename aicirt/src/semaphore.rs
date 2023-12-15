use std::ffi::CString;
use std::io;
use std::time::{Duration, Instant};

use anyhow::Result;

pub struct Semaphore {
    sem: *mut libc::sem_t,
}

impl Semaphore {
    fn last_error<T>() -> Result<T> {
        Err(io::Error::last_os_error().into())
    }

    pub fn new(name: &str, initial_value: u32, unlink: bool) -> Result<Self> {
        log::trace!("sem_open: {}", name);
        let c_name = CString::new(name).unwrap();
        if unlink {
            unsafe {
                libc::sem_unlink(c_name.as_ptr());
            };
        }
        let sem = unsafe { libc::sem_open(c_name.as_ptr(), libc::O_CREAT, 0o666, initial_value) };

        if sem.is_null() {
            return Self::last_error();
        }

        Ok(Self { sem })
    }

    pub fn wait(&self) -> Result<()> {
        let ret = unsafe { libc::sem_wait(self.sem) };
        if ret < 0 {
            return Self::last_error();
        }
        Ok(())
    }

    pub fn busy_wait(&self, wait_duration: &Duration) -> Result<()> {
        let deadline = Instant::now() + *wait_duration;
        loop {
            let ret = unsafe { libc::sem_trywait(self.sem) };
            if ret < 0 {
                #[cfg(target_os = "linux")]
                let last_error = unsafe { *libc::__errno_location() };
                #[cfg(not(target_os = "linux"))]
                let last_error = unsafe { *libc::__error() };
                if last_error == libc::EAGAIN {
                    if Instant::now() > deadline {
                        return self.wait();
                    } else {
                        // std::hint::spin_loop();
                        continue;
                    }
                } else {
                    return Self::last_error();
                }
            } else {
                return Ok(());
            }
        }
    }

    pub fn post(&self) -> Result<()> {
        let ret = unsafe { libc::sem_post(self.sem) };
        if ret < 0 {
            return Self::last_error();
        }
        Ok(())
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            libc::sem_close(self.sem);
        }
    }
}
