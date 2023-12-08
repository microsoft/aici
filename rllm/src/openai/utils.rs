use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn get_created_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time travel has occurred...")
        .as_secs()
}
