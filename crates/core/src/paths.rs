//! Path resolution: base dir (env-overridable), logs dir, registry path.

use std::path::PathBuf;

use crate::collectors::sanitize_id;

/// Base dir for data + logs. On the head node this is /home/sean/chronos;
/// fall back to cwd (e.g. when type-checking/running locally on a Mac).
/// `CHRONOS_HOME` takes precedence when set (tests / Mac runs).
pub fn base_dir() -> PathBuf {
    if let Ok(home) = std::env::var("CHRONOS_HOME") {
        if !home.trim().is_empty() {
            return PathBuf::from(home);
        }
    }
    let p = PathBuf::from("/home/sean/chronos");
    if p.is_dir() {
        p
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
}

pub fn logs_dir() -> PathBuf {
    base_dir().join("logs")
}

pub fn registry_path() -> PathBuf {
    base_dir().join("data").join("registry.json")
}

pub fn log_file_for(id: &str) -> PathBuf {
    logs_dir().join(format!("{}.log", sanitize_id(id)))
}
