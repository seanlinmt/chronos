//! chronos-core — shared domain core for the chronos workspace.
//!
//! Single source of truth for ALL domain logic shared by the TUI and HTTP
//! frontends:
//!
//! - registry (entry model, seed of the 5 cluster entries, load/seed/save)
//! - paths (`base_dir()` = `~/chronos`, `CHRONOS_HOME` env override)
//! - node metrics (meminfo, `df -B1 /`)
//! - GB10 per-PID GPU attribution (nvidia-smi compute apps + VmRSS + cgroup
//!   docker-scope matching + process_match)
//! - engine probes (tensorfold /health, vllm /v1/models + /metrics with HTTP
//!   status checks before body parsing, velogb10 /health, Prev-counter deltas)
//! - shared snapshot state + publisher loops (tokio watch channel) and the
//!   `--dump-state` JSON shape
//! - detached lifecycle (sh -c, process_group(0), stdin null, logs/<id>.log),
//!   start/stop/add/remove, drop-caches (both nodes), log tails

pub mod collectors;
pub mod lifecycle;
pub mod model;
pub mod paths;
pub mod probes;
pub mod registry;
pub mod state;

// ---------------------------------------------------------------------------
// Public API re-exports (consumed by chronos and chronos-http)
// ---------------------------------------------------------------------------

pub use collectors::{
    build_client, compact, gib, is_ai2_host, sanitize_id, sections,
};
pub use lifecycle::{
    add_entry, container_log_lines, drop_caches, entry_logs, log_file_tail, remove_entry,
    spawn_detached, start_docker, start_entry, stop_entry, update_entry, DropCachesResult,
};
pub use model::{
    AppState, DiskInfo, EntrySnapshot, GraceKind, NodeState, Prev, ProbeResult, RegistryEntry,
    AI2_LAN, AI_LAN, CHECK_TIMEOUT_SECS, HTTP_INTERVAL_SECS,
    HTTP_TIMEOUT_MS, SSH_BLOB_TIMEOUT_SECS, STARTING_GRACE_SECS, SYS_INTERVAL_SECS,
};
pub use paths::{base_dir, log_file_for, logs_dir, registry_path};
pub use registry::{load_or_seed_registry, save_registry, seed_registry};
pub use state::{run_dump_state, Collector, Shared};
pub use probes::CollectorState;
