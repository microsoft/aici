use std::{
    collections::HashMap,
    fmt::{Display, Write},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

#[derive(PartialEq, Eq)]
enum TimerState {
    Stopped,
    Running,
}

#[derive(Clone)]
pub struct TimerSet {
    inner: Arc<Mutex<TimerInner>>,
}

// #[derive(Clone)]
pub struct TimerRef {
    set: TimerSet,
    idx: usize,
}

struct TimerInner {
    timers: Vec<TimerImpl>,
}

struct TimerImpl {
    name: String,
    state: TimerState,
    start: Instant,
    elapsed: Duration,
    parent: Option<u32>,
    num: u32,
}

impl TimerSet {
    pub fn new() -> TimerSet {
        TimerSet {
            inner: Arc::new(Mutex::new(TimerInner { timers: Vec::new() })),
        }
    }

    pub fn new_timer(&self, name: &str) -> TimerRef {
        let mut inner = self.inner.lock().unwrap();
        let idx = inner.timers.len();
        inner.timers.push(TimerImpl::new(name));

        // recompute parents
        let all_tim: HashMap<String, u32> = HashMap::from_iter(
            inner
                .timers
                .iter()
                .enumerate()
                .map(|(idx, t)| (t.name.clone(), idx as u32)),
        );
        for timer in &mut inner.timers {
            let parts: Vec<&str> = timer.name.rsplitn(2, '.').collect();
            timer.parent = if parts.len() == 2 {
                all_tim.get(parts[1]).map(|&p| p)
            } else {
                None
            };
        }

        TimerRef {
            idx,
            set: self.clone(),
        }
    }

    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        for timer in &mut inner.timers {
            timer.reset();
        }
    }

    pub fn pp(&self) -> String {
        let inner = self.inner.lock().unwrap();
        inner.pp()
    }
}

fn spaces(n: usize) -> String {
    std::iter::repeat(' ').take(n).collect()
}

impl TimerInner {
    fn pp_one(&self, f: &mut String, ind: usize, myidx: usize) {
        let timer = &self.timers[myidx];
        let parent = timer.parent.map(|p| &self.timers[p as usize]);
        let name = match parent {
            Some(p) => &timer.name[p.name.len()..],
            None => timer.name.as_str(),
        };
        let perc = match parent {
            Some(p) => format!(
                "{:4.1}% ",
                100.0 * timer.elapsed.as_micros() as f64 / p.elapsed.as_micros() as f64
            ),
            None => String::new(),
        };
        write!(
            f,
            "{}{} {:8.3}ms (x{}) {}\n",
            spaces(ind),
            perc,
            timer.elapsed.as_micros() as f64 / 1000.0 / std::cmp::max(1, timer.num) as f64,
            timer.num,
            name,
        )
        .unwrap();
        for (idx, timer) in self.timers.iter().enumerate() {
            if timer.parent == Some(myidx as u32) {
                self.pp_one(f, ind + 4, idx);
            }
        }
    }

    fn pp(&self) -> String {
        let mut f = String::new();
        for (idx, timer) in self.timers.iter().enumerate() {
            if timer.parent.is_none() {
                self.pp_one(&mut f, 2, idx);
            }
        }
        f
    }
}

impl TimerRef {
    #[inline(never)]
    pub fn start(&self) {
        let mut inner = self.set.inner.lock().unwrap();
        inner.timers[self.idx].start();
    }

    #[inline(never)]
    pub fn stop(&self) {
        let mut inner = self.set.inner.lock().unwrap();
        inner.timers[self.idx].stop();
    }

    #[allow(dead_code)]
    pub fn set(&self) -> &TimerSet {
        &self.set
    }

    #[inline(always)]
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        self.start();
        let result = f();
        self.stop();
        result
    }
}

impl Display for TimerSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        for timer in &inner.timers {
            write!(f, "{}\n", timer)?;
        }
        Ok(())
    }
}

impl Display for TimerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.set.inner.lock().unwrap();
        write!(f, "{}", inner.timers[self.idx])
    }
}

impl Display for TimerImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let average = self.elapsed.as_micros() as f64 / 1000.0 / std::cmp::max(1, self.num) as f64;
        write!(f, "R_{}: {:.3}ms (x{})", self.name, average, self.num)
    }
}

impl TimerImpl {
    pub fn new(name: &str) -> TimerImpl {
        TimerImpl {
            name: name.to_string(),
            state: TimerState::Stopped,
            start: Instant::now(),
            elapsed: Duration::new(0, 0),
            parent: None,
            num: 0,
        }
    }

    #[inline(always)]
    fn start(&mut self) {
        assert!(self.state == TimerState::Stopped);
        self.start = Instant::now();
        self.state = TimerState::Running;
    }

    #[inline(always)]
    fn stop(&mut self) {
        assert!(self.state == TimerState::Running);
        self.elapsed += self.start.elapsed();
        self.num += 1;
        self.state = TimerState::Stopped;
    }

    fn reset(&mut self) {
        assert!(self.state == TimerState::Stopped);
        self.elapsed = Duration::new(0, 0);
        self.num = 0;
    }
}

#[macro_export]
macro_rules! with_timer {
    ($timer:expr, $block:block) => {{
        $timer.start();
        let r = $block;
        $timer.stop();
        r
    }};
    ($timer:expr, $block:expr) => {{
        $timer.start();
        let r = $block;
        $timer.stop();
        r
    }};
}
