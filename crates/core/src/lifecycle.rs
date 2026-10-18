//! Lifecycle actions: detached spawns (sh -c, process_group(0), stdin null,
//! output appended to logs/<id>.log), start/stop/add/remove, drop-caches on
//! both nodes, and log tails.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::collectors::{is_ai2_host, ssh_ai2};
use crate::model::{GraceKind, RegistryEntry};
use crate::paths::log_file_for;
use crate::registry::save_registry;
use crate::state::Shared;

/// Spawn a lifecycle command fully detached (setsid, stdin null, output
/// appended to the entry's log file) so quitting the TUI never kills children.
pub async fn spawn_detached(cwd: &str, cmd: &str, log_path: &Path) -> Result<(), String> {
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
        .map_err(|e| format!("spawn `{cmd}` in {cwd}: {e}"))?;
    Ok(())
}

fn find_entry(shared: &Shared, id: &str) -> Option<RegistryEntry> {
    shared.registry.read().unwrap().iter().find(|e| e.id == id).cloned()
}

pub async fn start_entry(id: &str, shared: &Arc<Shared>) -> Result<String, String> {
    let e = find_entry(shared, id).ok_or_else(|| format!("entry `{id}` not found"))?;
    let log = log_file_for(&e.id);
    match spawn_detached(&e.path, &e.start, &log).await {
        Ok(()) => {
            shared
                .grace
                .lock()
                .unwrap()
                .insert(e.id.clone(), (GraceKind::Starting, Instant::now()));
            Ok(format!("started `{}` (logs: {})", e.id, log.display()))
        }
        Err(err) => Err(format!("start failed: {err}")),
    }
}

pub async fn stop_entry(id: &str, shared: &Arc<Shared>) -> Result<String, String> {
    let e = find_entry(shared, id).ok_or_else(|| format!("entry `{id}` not found"))?;
    let log = log_file_for(&e.id);
    match spawn_detached(&e.path, &e.stop, &log).await {
        Ok(()) => {
            shared
                .grace
                .lock()
                .unwrap()
                .insert(e.id.clone(), (GraceKind::Stopping, Instant::now()));
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
