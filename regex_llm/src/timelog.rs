use std::{
    fmt,
    time::{Duration, Instant},
};

pub struct TimeLog {
    start: Instant,
    prev: Instant,
    times: Vec<(String, Duration)>,
}

impl TimeLog {
    pub fn new() -> Self {
        let now = Instant::now();
        TimeLog {
            start: now,
            prev: now,
            times: Vec::new(),
        }
    }
    pub fn save(&mut self, id: &str) {
        self.times.push((String::from(id), self.prev.elapsed()));
        self.prev = Instant::now();
    }
}

impl fmt::Display for TimeLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut copy = self.times.clone();
        copy.push((String::from("final"), self.prev.elapsed()));
        copy.push((String::from("TOTAL"), self.start.elapsed()));
        for (l, d) in &copy {
            write!(f, "{:8.1}ms {}\n", d.as_micros() as f64 / 100.0, l)?
        }
        Ok(())
    }
}
