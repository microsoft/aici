use perfcnt::{
    linux::{HardwareEventType, PerfCounterBuilderLinux},
    AbstractPerfCounter,
};

pub struct BenchmarkState {
    pub times: Vec<u64>,
    cnt: perfcnt::PerfCounter,
}

impl BenchmarkState {
    pub fn new() -> Self {
        let cnt = PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::Instructions)
            .finish()
            .expect("Failed to create counter");

        BenchmarkState { times: vec![], cnt }
    }

    pub fn measure(&mut self, f: impl FnOnce()) {
        let t0 = std::time::Instant::now();
        self.cnt.reset().unwrap();
        self.cnt.start().unwrap();
        f();
        self.cnt.stop().unwrap();
        let res = self.cnt.read().unwrap();
        let _res = t0.elapsed().as_nanos() as u64;
        self.times.push(res);
    }

    pub fn is_done(&self) -> bool {
        self.times.len() >= 20
    }

    pub fn print(&self) {
        let avg = self.times.iter().sum::<u64>() / self.times.len() as u64;
        let min = *self.times.iter().min().unwrap();
        let max = *self.times.iter().max().unwrap();
        let (t10, median, b10) = {
            let mut sorted = self.times.clone();
            sorted.sort();
            (
                sorted[sorted.len() / 10],
                sorted[sorted.len() / 2],
                sorted[sorted.len() * 9 / 10],
            )
        };
        let to_m = |x| x as f64 / 1_000_000.0;
        // println!("times: {:?}", self.times);
        println!(
            "Cycles: min:{:.3}-{:.3} med:{:.3} avg:{:.3} max:{:.3}-{:.3}",
            to_m(min),
            to_m(t10),
            to_m(median),
            to_m(avg),
            to_m(b10),
            to_m(max)
        );
    }
}
