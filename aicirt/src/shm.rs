use anyhow::{anyhow, ensure, Result};
use std::{
    ffi::CString,
    io, ptr,
    sync::atomic::{AtomicU32, Ordering},
};

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

pub struct ShmAllocator {
    pub shm: Shm,
}

#[repr(C)]
struct ShmAllocatorHeader {
    pub magic: u32,
    pub elt_size: u32,
    pub num_elts: u32,
    pub elt_type: u32,
}

impl ShmAllocator {
    const MAGIC: u32 = 0xf01291b7;
    const HEADER_SIZE: usize = 64;
    const FREE: u32 = u32::MAX;

    pub fn new(shm: Shm, elt_size: usize, elt_type: u32) -> Self {
        let mut s = Self::new_no_init(shm);
        s.init(elt_size, elt_type);
        s
    }

    pub fn new_no_init(shm: Shm) -> Self {
        Self { shm }
    }

    fn get_header(&self) -> &mut ShmAllocatorHeader {
        &mut self.shm.slice_at_byte_offset::<ShmAllocatorHeader>(0, 1)[0]
    }

    fn alloc_table(&self) -> &mut [AtomicU32] {
        self.shm
            .slice_at_byte_offset(Self::HEADER_SIZE, self.num_elts())
    }

    pub fn num_elts(&self) -> usize {
        self.get_header().num_elts as usize
    }

    pub fn elt_size(&self) -> usize {
        self.get_header().elt_size as usize
    }

    pub fn elt_type(&self) -> u32 {
        self.get_header().elt_type
    }

    fn init(&mut self, elt_size: usize, elt_type: u32) {
        assert!(elt_size <= 0x1000_0000);
        assert!(std::mem::size_of::<ShmAllocatorHeader>() <= Self::HEADER_SIZE);
        let header = self.get_header();
        header.magic = Self::MAGIC;
        header.elt_size = elt_size as u32;
        header.num_elts = (self.shm.size as u32 - 128) / (elt_size + 8) as u32;
        header.elt_type = elt_type;
        assert!(header.num_elts > 0);
        self.alloc_table()
            .iter_mut()
            .for_each(|x| x.store(Self::FREE, Ordering::SeqCst));
    }

    fn data_off(&self) -> usize {
        Self::HEADER_SIZE + ((self.num_elts() + 3) & !3) * std::mem::size_of::<AtomicU32>()
    }

    pub fn free(&self, max_offset: usize, condition: impl Fn(u32) -> bool) {
        let table = self.alloc_table();
        let max_ent = std::cmp::min(
            table.len(),
            (max_offset - self.data_off()) / self.elt_size() + 1,
        );
        for i in 0..max_ent {
            let e = table[i].load(Ordering::Relaxed);
            if e != 0 && condition(e) {
                table[i].store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn slice_at_byte_offset<T>(&self, off: usize, num_elts: usize) -> &'static mut [T] {
        assert!(num_elts * std::mem::size_of::<T>() <= self.elt_size());
        self.shm.slice_at_byte_offset(off, num_elts)
    }

    pub fn alloc(&self, client_id: u32) -> Result<usize> {
        assert!(client_id != Self::FREE);
        let table = self.alloc_table();
        for i in 0..table.len() {
            let mut current = table[i].load(Ordering::Relaxed);
            while current == 0 {
                if table[i]
                    .compare_exchange_weak(current, client_id, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let elt_size = self.elt_size();
                    let elt_off = self.data_off() + i * elt_size;
                    assert!(elt_off + elt_size <= self.shm.size);
                    return Ok(elt_off);
                }
                current = table[i].load(Ordering::Relaxed);
            }
        }
        Err(anyhow!("no free slots"))
    }
}
