use std::time::{Duration, Instant};

use wasmtime_wasi::{HostMonotonicClock, HostWallClock};

#[derive(Debug, Clone)]
pub struct BoundedResolutionClock {
    resolution: Duration,
    initial: Instant,
}

impl BoundedResolutionClock {
    pub fn new(resolution: Duration) -> Self {
        BoundedResolutionClock {
            resolution,
            initial: Instant::now(),
        }
    }
}

impl HostMonotonicClock for BoundedResolutionClock {
    fn resolution(&self) -> u64 {
        self.resolution.as_nanos() as u64
    }

    fn now(&self) -> u64 {
        let now = std::time::Instant::now();
        let nanos = now.duration_since(self.initial).as_nanos() as u64;
        let res = self.resolution.as_nanos() as u64;
        let nanos = if res > 0 { nanos / res * res } else { nanos };
        nanos as u64
    }
}

impl HostWallClock for BoundedResolutionClock {
    fn resolution(&self) -> Duration {
        self.resolution
    }

    fn now(&self) -> Duration {
        let now = std::time::SystemTime::now();
        let nanos = now
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let res = self.resolution.as_nanos() as u64;
        let nanos = if res > 0 { nanos / res * res } else { nanos };
        Duration::from_nanos(nanos)
    }
}
