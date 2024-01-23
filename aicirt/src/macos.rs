use anyhow::{anyhow, Result};
use std::{sync::atomic::AtomicU32, time::Duration};
use ulock_sys::{
    __ulock_wait, __ulock_wake, darwin19::UL_COMPARE_AND_WAIT_SHARED, ULF_NO_ERRNO, ULF_WAKE_ALL,
};

pub trait AsFutex {
    fn as_futex(&self) -> &Futex;
}

impl AsFutex for AtomicU32 {
    #[must_use]
    #[inline]
    fn as_futex(&self) -> &Futex {
        unsafe { std::mem::transmute(self) }
    }
}

#[repr(transparent)]
pub struct Futex {
    pub value: AtomicU32,
}

impl Futex {
    fn wait_core(&self, expected_value: u32, micros: u32) -> Result<()> {
        let r = unsafe {
            __ulock_wait(
                UL_COMPARE_AND_WAIT_SHARED | ULF_NO_ERRNO,
                self.value.as_ptr() as *mut libc::c_void,
                expected_value as u64,
                micros,
            )
        };

        if r >= 0 {
            Ok(())
        } else {
            // TODO: can copy errors from https://github.com/ziglang/zig/blob/9e684e8d1af39904055abe64a9afda69a3d44a59/lib/std/Thread/Futex.zig#L192
            Err(anyhow!("__ulock_wait failed: {}", r))
        }
    }

    /// Wait until this futex is awoken by a `wake` call.
    /// The thread will only be sent to sleep if the futex's value matches the
    /// expected value.
    pub fn wait(&self, expected_value: u32) -> Result<()> {
        self.wait_core(expected_value, 0)
    }

    /// Wait until this futex is awoken by a `wake` call, or until the timeout expires.
    /// The thread will only be sent to sleep if the futex's value matches the
    /// expected value.
    pub fn wait_for(&self, expected_value: u32, timeout: Duration) -> Result<()> {
        if timeout >= Duration::from_micros(u32::MAX as u64) {
            self.wait_core(expected_value, 0)
        } else {
            self.wait_core(expected_value, timeout.as_micros() as u32)
        }
    }

    /// Wake up `n` waiters.
    pub fn wake(&self, _n: i32) -> i32 {
        loop {
            let r = unsafe {
                __ulock_wake(
                    UL_COMPARE_AND_WAIT_SHARED | ULF_NO_ERRNO | ULF_WAKE_ALL,
                    self.value.as_ptr() as *mut libc::c_void,
                    0,
                )
            };
            if r == -libc::ENOENT {
                return 0;
            }
            if r >= 0 {
                return 1;
            }
        }
    }
}
