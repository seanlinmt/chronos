//! Lifecycle actions: detached spawns (sh -c, process_group(0), stdin null,
//! output appended to logs/<id>.log), start/stop/add/remove, drop-caches on
//! both nodes, and log tails.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::collectors::{
    is_ai2_host, parse_throttle_stat, read_throttle_stat, ssh_ai2, GpuStat,
};
use crate::model::{GraceKind, RegistryEntry, SSH_BLOB_TIMEOUT_SECS};
use crate::paths::log_file_for;
use crate::registry::save_registry;
use crate::state::Shared;

/// Spawn a lifecycle command fully detached (setsid, stdin null, output
/// appended to the entry's log file) so quitting the TUI never kills children.
/// Returns the child so callers can reap it (and react to early failure).
pub async fn spawn_detached(cwd: &str, cmd: &str, log_path: &Path) -> Result<tokio::process::Child, String> {
    use std::process::Stdio;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open log {}: {e}", log_path.display()))?;
    let stderr = file
        .try_clone()
        .map_err(|e| format!("clone log handle: {e}"))?;
    tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(cwd)
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|e| format!("spawn `{cmd}` in {cwd}: {e}"))
}

/// Reap a detached child in the background (avoids zombies). When
/// `clear_starting_grace` is set, a non-zero exit removes the entry's
/// `Starting` grace so the UI flips to STOPPED immediately instead of
/// waiting out the full grace window after a fast-failing launcher.
fn watch_detached(
    mut child: tokio::process::Child,
    id: String,
    shared: &Arc<Shared>,
    clear_starting_grace: bool,
) {
    let sh = shared.clone();
    tokio::spawn(async move {
        if let Ok(status) = child.wait().await {
            if clear_starting_grace && !status.success() {
                sh.grace.lock().unwrap().remove(&id);
                sh.publish();
            }
        }
    });
}

fn find_entry(shared: &Shared, id: &str) -> Option<RegistryEntry> {
    shared.registry.read().unwrap().iter().find(|e| e.id == id).cloned()
}

pub async fn start_entry(id: &str, shared: &Arc<Shared>) -> Result<String, String> {
    let e = find_entry(shared, id).ok_or_else(|| format!("entry `{id}` not found"))?;
    let log = log_file_for(&e.id);
    match spawn_detached(&e.path, &e.start, &log).await {
        Ok(child) => {
            shared
                .grace
                .lock()
                .unwrap()
                .insert(e.id.clone(), (GraceKind::Starting, Instant::now()));
            watch_detached(child, e.id.clone(), shared, true);
            Ok(format!("started `{}` (logs: {})", e.id, log.display()))
        }
        Err(err) => Err(format!("start failed: {err}")),
    }
}

pub async fn stop_entry(id: &str, shared: &Arc<Shared>) -> Result<String, String> {
    let e = find_entry(shared, id).ok_or_else(|| format!("entry `{id}` not found"))?;
    let log = log_file_for(&e.id);
    match spawn_detached(&e.path, &e.stop, &log).await {
        Ok(child) => {
            shared
                .grace
                .lock()
                .unwrap()
                .insert(e.id.clone(), (GraceKind::Stopping, Instant::now()));
            watch_detached(child, e.id.clone(), shared, false);
            Ok(format!("stopping `{}` (logs: {})", e.id, log.display()))
        }
        Err(err) => Err(format!("stop failed: {err}")),
    }
}

pub fn add_entry(entry: RegistryEntry, shared: &Arc<Shared>) -> Result<String, String> {
    let mut reg = shared.registry.write().unwrap();
    if reg.iter().any(|e| e.id == entry.id) {
        return Err(format!("entry `{}` already exists", entry.id));
    }
    reg.push(entry.clone());
    save_registry(&reg).map(|_| format!("added `{}` to registry", entry.id))
}

/// Removes the registry record only — never deletes files.
pub fn remove_entry(id: &str, shared: &Arc<Shared>) -> Result<String, String> {
    let mut reg = shared.registry.write().unwrap();
    let before = reg.len();
    reg.retain(|e| e.id != id);
    if reg.len() == before {
        return Err(format!("entry `{id}` not found"));
    }
    save_registry(&reg).map(|_| format!("removed `{id}` from registry (files kept)"))
}

pub fn update_entry(old_id: &str, entry: RegistryEntry, shared: &Arc<Shared>) -> Result<String, String> {
    let mut reg = shared.registry.write().unwrap();
    let pos = reg
        .iter()
        .position(|e| e.id == old_id)
        .ok_or_else(|| format!("entry `{old_id}` not found"))?;

    if entry.id != old_id && reg.iter().any(|e| e.id == entry.id) {
        return Err(format!("entry `{}` already exists", entry.id));
    }

    if entry.id != old_id {
        {
            let mut states = shared.states.write().unwrap();
            if let Some(s) = states.remove(old_id) {
                states.insert(entry.id.clone(), s);
            }
        }
        {
            let mut mem = shared.mem.write().unwrap();
            if let Some(m) = mem.remove(old_id) {
                mem.insert(entry.id.clone(), m);
            }
        }
        {
            let mut probes = shared.probes.write().unwrap();
            if let Some(p) = probes.remove(old_id) {
                probes.insert(entry.id.clone(), p);
            }
        }
        {
            let mut grace = shared.grace.lock().unwrap();
            if let Some(g) = grace.remove(old_id) {
                grace.insert(entry.id.clone(), g);
            }
        }
    }

    reg[pos] = entry.clone();
    save_registry(&reg).map(|_| format!("updated `{}` in registry", entry.id))
}


/// Result of a drop-caches pass; `ai`/`ai2` are pre-formatted status strings
/// ("ok", "failed (...)", "error: ...", "timed out").
pub struct DropCachesResult {
    pub ok: bool,
    pub ai: String,
    pub ai2: String,
}

pub async fn drop_caches() -> DropCachesResult {
    let local = tokio::process::Command::new("sh")
        .args(["-c", "sudo -n sync && sudo -n sysctl -w vm.drop_caches=3"])
        .output()
        .await;
    let remote = tokio::time::timeout(
        Duration::from_secs(5),
        ssh_ai2("sudo -n sync && sudo -n sysctl -w vm.drop_caches=3"),
    )
    .await;
    let ai = match &local {
        Ok(o) if o.status.success() => "ok".to_string(),
        Ok(_) => "failed (needs passwordless sudo)".to_string(),
        Err(e) => format!("error: {e}"),
    };
    let ai2 = match remote {
        Ok(Ok(_)) => "ok".to_string(),
        Ok(Err(e)) => format!("failed ({e})"),
        Err(_) => "timed out".to_string(),
    };
    let ok = ai == "ok" && ai2 == "ok";
    DropCachesResult { ok, ai, ai2 }
}

/// Start the systemd docker service on a specific node (`ai` or `ai2`).
pub async fn start_docker(node: &str) -> Result<String, String> {
    match node {
        "ai" => {
            let out = tokio::time::timeout(
                Duration::from_secs(10),
                tokio::process::Command::new("sh")
                    .args(["-c", "sudo -n systemctl start docker"])
                    .output(),
            )
            .await
            .map_err(|_| "timed out starting docker on ai".to_string())?
            .map_err(|e| format!("failed to run systemctl: {e}"))?;

            if out.status.success() {
                Ok("docker started on ai".to_string())
            } else {
                let err = String::from_utf8_lossy(&out.stderr);
                Err(format!("failed to start docker on ai: {}", err.trim()))
            }
        }
        "ai2" => {
            tokio::time::timeout(
                Duration::from_secs(10),
                ssh_ai2("sudo -n systemctl start docker"),
            )
            .await
            .map_err(|_| "timed out starting docker on ai2".to_string())?
            .map_err(|e| format!("failed to start docker on ai2: {e}"))?;

            Ok("docker started on ai2".to_string())
        }
        other => Err(format!("unknown node `{other}`")),
    }
}

// =============================================================================
// GPU THROTTLE BASELINE
// =============================================================================
//
// GPU clocks-event counters CANNOT be cleared from userspace on this driver
// (`nvidia-smi -rgc` leaves them unchanged; a real clear needs a driver reload
// or `-r`, both too invasive on a live inference host). So chronos keeps its
// OWN baseline: a persisted snapshot of the five µs counters, with the UI
// showing deltas since that snapshot. The GPU counters themselves are never
// reset.

/// The five GPU throttle counters that can be baselined (µs, accumulated since
/// the driver last loaded).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CounterSnapshot {
    #[serde(default)]
    pub sw_power_cap_us: Option<u64>,
    #[serde(default)]
    pub sw_thermal_us: Option<u64>,
    #[serde(default)]
    pub hw_thermal_us: Option<u64>,
    #[serde(default)]
    pub hw_power_brake_us: Option<u64>,
    #[serde(default)]
    pub sync_boost_us: Option<u64>,
}

impl CounterSnapshot {
    /// Extract the counter set from a parsed PERFORMANCE dump.
    fn from_stat(s: &GpuStat) -> CounterSnapshot {
        CounterSnapshot {
            sw_power_cap_us: s.sw_power_cap_us,
            sw_thermal_us: s.sw_thermal_us,
            hw_thermal_us: s.hw_thermal_us,
            hw_power_brake_us: s.hw_power_brake_us,
            sync_boost_us: s.sync_boost_us,
        }
    }

    /// True when no counter was available (nvidia-smi missing/failed).
    pub fn is_empty(&self) -> bool {
        self.sw_power_cap_us.is_none()
            && self.sw_thermal_us.is_none()
            && self.hw_thermal_us.is_none()
            && self.hw_power_brake_us.is_none()
            && self.sync_boost_us.is_none()
    }

    /// Counter deltas since `base`. A field stays `None` when either side is
    /// unavailable (no baseline, or the current counter is missing). A REWOUND
    /// counter (`current < baseline`, i.e. the driver reloaded) is also `None`
    /// rather than a clamped `0`: a false "0 s since reset" would read as an
    /// all-clear when the counters actually restarted.
    pub fn delta_since(&self, base: &CounterSnapshot) -> CounterSnapshot {
        fn d(cur: Option<u64>, base: Option<u64>) -> Option<u64> {
            match (cur, base) {
                (Some(c), Some(b)) if c >= b => Some(c - b),
                _ => None,
            }
        }
        CounterSnapshot {
            sw_power_cap_us: d(self.sw_power_cap_us, base.sw_power_cap_us),
            sw_thermal_us: d(self.sw_thermal_us, base.sw_thermal_us),
            hw_thermal_us: d(self.hw_thermal_us, base.hw_thermal_us),
            hw_power_brake_us: d(self.hw_power_brake_us, base.hw_power_brake_us),
            sync_boost_us: d(self.sync_boost_us, base.sync_boost_us),
        }
    }
}

/// A chronos-side throttle baseline. `ts` is the RFC3339 capture time (stored
/// as a string so no extra serde feature is needed for chrono).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottleBaseline {
    #[serde(default)]
    pub ts: String,
    #[serde(default)]
    pub ai: CounterSnapshot,
    #[serde(default)]
    pub ai2: CounterSnapshot,
}

/// Outcome of a baseline capture. `ai`/`ai2` report whether THAT node's
/// counters were actually read in this pass; `persisted` reports whether the
/// snapshot reached disk.
#[derive(Debug, Clone)]
pub struct ThrottleResetResult {
    pub baseline: ThrottleBaseline,
    pub ai: bool,
    pub ai2: bool,
    pub persisted: bool,
    pub persist_error: Option<String>,
}

/// `<CHRONOS_HOME or ~/chronos>/data/throttle-baseline.json`.
fn throttle_baseline_path() -> std::path::PathBuf {
    crate::paths::base_dir()
        .join("data")
        .join("throttle-baseline.json")
}

/// Persist the baseline atomically (write `<file>.tmp` in the same directory,
/// then rename — atomic on ext4) and PROPAGATE errors: a baseline that never
/// reached disk disappears on the next restart, so the caller must not report
/// success for it.
fn save_throttle_baseline(b: &ThrottleBaseline) -> std::io::Result<()> {
    let path = throttle_baseline_path();
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("baseline path has no parent"))?;
    std::fs::create_dir_all(dir)?;
    let text = serde_json::to_string_pretty(b).map_err(std::io::Error::other)?;
    let tmp = dir.join("throttle-baseline.json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, &path)
}

/// Load the persisted baseline (None when absent/unreadable). Used at startup
/// so `since_reset_us` survives a service restart.
pub fn load_throttle_baseline() -> Option<ThrottleBaseline> {
    let text = std::fs::read_to_string(throttle_baseline_path()).ok()?;
    serde_json::from_str(&text).ok()
}

/// A capture that produced no usable counters (nvidia-smi missing, timed out,
/// or the dump had no recognised throttle section) is a failure, not a snapshot
/// of "everything unknown".
fn usable(c: Option<CounterSnapshot>) -> Option<CounterSnapshot> {
    c.filter(|x| !x.is_empty())
}

/// Capture a NEW throttle baseline for both nodes and persist it.
///
/// Reads `ai` locally and `ai2` over the existing `ssh_ai2` helper. Errors only
/// when counters are unavailable on BOTH nodes. A node whose read failed is
/// reported as `false` in [`ThrottleResetResult`] and — crucially — its
/// PREVIOUS baseline values are carried forward if one exists, so a transient
/// ssh/nvidia-smi failure never overwrites a good snapshot with all-`None`
/// (which would silently null every `since_reset_us` for that node forever).
pub async fn reset_throttle_baseline() -> Result<ThrottleResetResult, String> {
    let ai_new = usable(
        read_throttle_stat()
            .await
            .map(|s| CounterSnapshot::from_stat(&s)),
    );

    // `grep -v '^==='` drops nvidia-smi's banner; harmless here but keeps the
    // remote payload consistent with AI2_BLOB.
    let ai2_new = usable(match tokio::time::timeout(
        Duration::from_secs(SSH_BLOB_TIMEOUT_SECS),
        ssh_ai2("nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -v '^===' || true"),
    )
    .await
    {
        Ok(Ok(out)) => parse_throttle_stat(&out).map(|s| CounterSnapshot::from_stat(&s)),
        _ => None,
    });

    if ai_new.is_none() && ai2_new.is_none() {
        return Err("GPU throttle counters unavailable on both ai and ai2".to_string());
    }

    // Capture the per-node outcome BEFORE the options are consumed below, so
    // the report cannot disagree with what was actually read.
    let ai_captured = ai_new.is_some();
    let ai2_captured = ai2_new.is_some();

    let previous = load_throttle_baseline();
    let ai = ai_new
        .or_else(|| previous.as_ref().map(|p| p.ai.clone()))
        .unwrap_or_default();
    let ai2 = ai2_new
        .or_else(|| previous.as_ref().map(|p| p.ai2.clone()))
        .unwrap_or_default();

    let baseline = ThrottleBaseline {
        ts: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        ai,
        ai2,
    };
    let (persisted, persist_error) = match save_throttle_baseline(&baseline) {
        Ok(()) => (true, None),
        Err(e) => (false, Some(format!("persist failed: {e}"))),
    };

    Ok(ThrottleResetResult {
        ai: ai_captured,
        ai2: ai2_captured,
        baseline,
        persisted,
        persist_error,
    })
}

/// Tail the entry's own log file (last `n` lines).
pub async fn log_file_tail(id: &str, n: usize) -> Result<Vec<String>, String> {
    let log = log_file_for(id);
    let data = tokio::fs::read_to_string(&log)
        .await
        .map_err(|e| format!("cannot read log: {e}"))?;
    let all: Vec<&str> = data.lines().collect();
    let skip = all.len().saturating_sub(n);
    Ok(all.into_iter().skip(skip).map(|s| s.to_string()).collect())
}

/// `docker logs --tail n <container>` — ssh for ai2-hosted entries.
/// Returns stdout lines (+ stderr when non-empty), or an error message.
pub async fn container_log_lines(host: &str, container: &str, n: usize) -> Result<Vec<String>, String> {
    let cmd = if is_ai2_host(host) {
        format!("ssh -o BatchMode=yes -o ConnectTimeout=3 ai2 'docker logs --tail {n} {container}'")
    } else {
        format!("docker logs --tail {n} {container}")
    };
    let out = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::process::Command::new("sh").arg("-c").arg(&cmd).output().await
    })
    .await;
    match out {
        Ok(Ok(o)) => {
            let mut lines: Vec<String> = String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(|s| s.to_string())
                .collect();
            let err = String::from_utf8_lossy(&o.stderr);
            if !err.trim().is_empty() {
                lines.push(err.to_string());
            }
            Ok(lines)
        }
        _ => Err("docker logs failed/timed out".into()),
    }
}

/// TUI-style combined log view: file tail (200) + docker logs (50) per
/// container. Returns pre-formatted display lines.
pub async fn entry_logs(id: &str) -> Result<Vec<String>, String> {
    let e = {
        let path = log_file_for(id);
        // Confirm the entry exists by looking for its log path derivation only
        // if we can; the caller (Collector) already checked existence.
        let _ = path;
        crate::registry::load_or_seed_registry()
            .into_iter()
            .find(|x| x.id == id)
            .ok_or_else(|| format!("entry `{id}` not found"))?
    };
    let mut lines = Vec::new();
    let log = log_file_for(&e.id);
    lines.push(format!("── tail {} ──", log.display()));
    match tokio::fs::read_to_string(&log).await {
        Ok(data) => {
            let all: Vec<&str> = data.lines().collect();
            let skip = all.len().saturating_sub(200);
            lines.extend(all.into_iter().skip(skip).map(|s| s.to_string()));
        }
        Err(err) => lines.push(format!("(cannot read log: {err})")),
    }
    for c in &e.containers {
        lines.push(format!("── docker logs --tail 50 {c} ──"));
        match container_log_lines(&e.host, c, 50).await {
            Ok(mut out) => lines.append(&mut out),
            Err(err) => lines.push(format!("({err})")),
        }
    }
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;
    use tokio::sync::watch;
    use crate::model::AppState;

    fn mock_shared(entries: Vec<RegistryEntry>) -> Arc<Shared> {
        let (tx, _rx) = watch::channel(Arc::new(AppState::default()));
        Arc::new(Shared {
            tx,
            registry: RwLock::new(entries),
            nodes: RwLock::new(Default::default()),
            mem: RwLock::new(Default::default()),
            probes: RwLock::new(Default::default()),
            states: RwLock::new(Default::default()),
            grace: std::sync::Mutex::new(Default::default()),
            cpu_prev: std::sync::Mutex::new(Default::default()),
            throttle_baseline: std::sync::RwLock::new(None),
        })
    }

    fn sample_entry(id: &str, port: u16) -> RegistryEntry {
        RegistryEntry {
            id: id.into(),
            path: format!("/path/{id}"),
            start: "./start.sh".into(),
            stop: "./stop.sh".into(),
            host: "192.168.20.63".into(),
            port,
            slot: "primary-llm".into(),
            engine: "vllm".into(),
            containers: vec![],
            process_match: None,
        }
    }

    #[test]
    fn test_update_entry_fields() {
        let tmp = std::env::temp_dir().join(format!("chronos_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        unsafe { std::env::set_var("CHRONOS_HOME", tmp.to_str().unwrap()); }

        let e1 = sample_entry("e1", 8888);
        let shared = mock_shared(vec![e1]);

        let mut updated = sample_entry("e1", 8899);
        updated.path = "/new/path".into();
        let res = update_entry("e1", updated.clone(), &shared);
        assert!(res.is_ok(), "{res:?}");

        let reg = shared.registry.read().unwrap();
        assert_eq!(reg.len(), 1);
        assert_eq!(reg[0].port, 8899);
        assert_eq!(reg[0].path, "/new/path");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_update_entry_rename_and_conflict() {
        let tmp = std::env::temp_dir().join(format!("chronos_test_rename_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        unsafe { std::env::set_var("CHRONOS_HOME", tmp.to_str().unwrap()); }

        let e1 = sample_entry("e1", 8888);
        let e2 = sample_entry("e2", 8889);
        let shared = mock_shared(vec![e1, e2]);
        shared.states.write().unwrap().insert("e1".into(), "RUNNING".into());

        // Attempt to rename e1 to e2 -> conflict
        let conflicting = sample_entry("e2", 8888);
        let res = update_entry("e1", conflicting, &shared);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("already exists"));

        // Rename e1 to e3 -> success
        let renamed = sample_entry("e3", 8888);
        let res = update_entry("e1", renamed, &shared);
        assert!(res.is_ok());

        let reg = shared.registry.read().unwrap();
        assert_eq!(reg.len(), 2);
        assert_eq!(reg[0].id, "e3");
        assert_eq!(reg[1].id, "e2");

        let states = shared.states.read().unwrap();
        assert_eq!(states.get("e3").map(String::as_str), Some("RUNNING"));
        assert_eq!(states.get("e1"), None);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_update_entry_not_found() {
        let tmp = std::env::temp_dir().join(format!("chronos_test_nf_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        unsafe { std::env::set_var("CHRONOS_HOME", tmp.to_str().unwrap()); }

        let shared = mock_shared(vec![]);
        let e = sample_entry("missing", 8888);
        let res = update_entry("missing", e, &shared);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("not found"));

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
