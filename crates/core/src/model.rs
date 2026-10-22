//! Shared data model: constants, registry entry, snapshot types, probe results.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};

// =============================================================================
// CONSTANTS
// =============================================================================

pub const AI2_LAN: &str = "192.168.20.229";
pub const AI_LAN: &str = "192.168.20.63";
pub const HTTP_TIMEOUT_MS: u64 = 1500;
pub const HTTP_INTERVAL_SECS: u64 = 2;
pub const SYS_INTERVAL_SECS: u64 = 3;
pub const CHECK_TIMEOUT_SECS: u64 = 2; // per storage/system check
pub const SSH_BLOB_TIMEOUT_SECS: u64 = 6; // one combined ssh batch for ai2
pub const STARTING_GRACE_SECS: u64 = 300; // STARTING until endpoint answers (or this expires)

// =============================================================================
// DATA MODEL
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    pub id: String,
    pub path: String,
    #[serde(default)]
    pub start: String,
    #[serde(default)]
    pub stop: String,
    pub host: String,
    pub port: u16,
    #[serde(default)]
    pub slot: String,
    #[serde(default = "default_engine")]
    pub engine: String,
    #[serde(default)]
    pub containers: Vec<String>,
    #[serde(default)]
    pub process_match: Option<String>,
}

pub fn default_engine() -> String {
    "vllm".to_string()
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StorageChecks {
    pub ai_findmnt: bool,
    pub ai_mdstat: bool,
    pub ai_stat: bool,
    pub ai2_findmnt: bool,
    pub ai2_stat: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StorageState {
    pub online: bool,
    pub fstype: Option<String>,
    pub reason: Option<String>,
    pub total_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub avail_bytes: Option<u64>,
    pub checks: StorageChecks,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct DiskInfo {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub avail_bytes: u64,
    pub use_pct: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct NodeState {
    pub mem_total_bytes: Option<u64>,
    pub mem_available_bytes: Option<u64>,
    pub mem_used_bytes: Option<u64>,
    pub mem_cache_bytes: Option<u64>,
    pub swap_total_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub disk_root: Option<DiskInfo>,
    /// Aggregate CPU busy percentage from /proc/stat deltas (None without a
    /// previous sample).
    #[serde(default)]
    pub cpu_load_pct: Option<f64>,
    /// Hottest valid /sys/class/thermal thermal_zone* sensor (CPU/SoC, °C).
    #[serde(default)]
    pub cpu_temp_c: Option<f64>,
    /// nvidia-smi GPU temperature (°C).
    #[serde(default)]
    pub gpu_temp_c: Option<f64>,
    /// nvidia-smi GPU utilization percentage (0–100, `None` when the driver
    /// reports `[N/A]` or nvidia-smi is unavailable).
    #[serde(default)]
    pub gpu_load_pct: Option<f64>,
}

/// One entry in the published JSON state (shape fixed by spec).
#[derive(Debug, Clone, Serialize)]
pub struct EntrySnapshot {
    pub id: String,
    pub path: String,
    pub slot: String,
    pub engine: String,
    pub host: String,
    pub port: u16,
    pub state: String,
    pub containers: Vec<String>,
    pub served_model: Option<String>,
    pub status_detail: Option<String>,
    pub mem_ai_bytes: u64,
    pub mem_ai2_bytes: u64,
    pub mem_total_bytes: u64,
    pub concurrent_requests: Option<u64>,
    pub queued_requests: Option<u64>,
    pub live_tok_per_sec: Option<f64>,
    pub lifetime_decode_tok_per_sec: Option<f64>,
    pub total_tokens_generated: Option<u64>,
    pub prompt_tokens_total: Option<u64>,
    pub cached_tokens_total: Option<u64>,
}

/// Top-level JSON document (also the TUI snapshot).
#[derive(Debug, Clone, Serialize, Default)]
pub struct AppState {
    pub timestamp: String,
    pub storage: StorageState,
    pub nodes: BTreeMap<String, Option<NodeState>>,
    pub entries: Vec<EntrySnapshot>,
}

/// Rolling counters kept between polls to compute deltas.
#[derive(Debug, Clone, Default)]
pub struct Prev {
    pub total: Option<u64>,
    pub decode_s: Option<f64>,
    pub at: Option<Instant>,
}

#[derive(Debug, Clone, Default)]
pub struct ProbeResult {
    pub ok: bool,
    pub detail: Option<String>,
    pub served_model: Option<String>,
    pub concurrent: Option<u64>,
    pub queued: Option<u64>,
    pub live_tok_per_sec: Option<f64>,
    pub lifetime_decode: Option<f64>,
    pub total_tokens: Option<u64>,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraceKind {
    Starting,
    Stopping,
}
