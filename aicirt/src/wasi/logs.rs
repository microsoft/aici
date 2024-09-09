use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use wasmtime_wasi::{async_trait, HostOutputStream, StdoutStream, StreamError, Subscribe};

#[derive(Debug, Clone)]
pub struct BoundedLogPipe {
    pub capacity: usize,
    pub buffer: Arc<Mutex<bytes::BytesMut>>,
}

impl BoundedLogPipe {
    pub fn new(capacity: usize) -> Self {
        BoundedLogPipe {
            capacity,
            buffer: std::sync::Arc::new(std::sync::Mutex::new(bytes::BytesMut::new())),
        }
    }

    pub fn contents(&self) -> bytes::Bytes {
        self.buffer.lock().unwrap().clone().freeze()
    }

    /// Drain the contents of the buffer, emptying it and returning the former contents.
    pub fn drain_contents(&self) -> bytes::Bytes {
        let mut buf = self.buffer.lock().unwrap();
        std::mem::replace(&mut *buf, bytes::BytesMut::new()).freeze()
    }
}

impl HostOutputStream for BoundedLogPipe {
    fn write(&mut self, bytes: bytes::Bytes) -> Result<(), StreamError> {
        let mut buf = self.buffer.lock().unwrap();
        if bytes.len() > self.capacity - buf.len() {
            return Err(StreamError::Trap(anyhow!(
                "write beyond capacity of BoundedLogPipe"
            )));
        }
        buf.extend_from_slice(bytes.as_ref());
        // Always ready for writing
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StreamError> {
        // This stream is always flushed
        Ok(())
    }

    fn check_write(&mut self) -> Result<usize, StreamError> {
        let consumed = self.buffer.lock().unwrap().len();
        if consumed < self.capacity {
            Ok(self.capacity - consumed)
        } else {
            // Since the buffer is full, no more bytes will ever be written
            Err(StreamError::Closed)
        }
    }
}

#[async_trait]
impl Subscribe for BoundedLogPipe {
    async fn ready(&mut self) {}
}


impl StdoutStream for BoundedLogPipe {
    fn stream(&self) -> Box<dyn HostOutputStream> {
        Box::new(self.clone())
    }

    fn isatty(&self) -> bool {
        true // Otherwise terminal_stdout
    }
}
