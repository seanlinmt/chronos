//! spark-deck-http — 2-node NVIDIA DGX Spark (GB10) cluster dashboard backend.
//!
//! Single-binary daemon. Binds 0.0.0.0:8090, serves the embedded UI plus a JSON
//! API, and runs three background loops (storage, metrics, telemetry) that
//! probe the local node (`ai`) and the remote worker (`ai2` via ssh).
//!
//! Paths are relative (resolved from WorkingDirectory):
//!   data/registry.json  — persisted entry registry
//!   logs/<id>.log       — start/stop command output

use std::collections::{HashMap, HashSet};
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::time::{sleep, timeout};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const BIND_ADDR: &str = "0.0.0.0:8090";
const LOCAL_NODE: &str = "ai";
const REMOTE_NODE: &str = "ai2";
const SSH_BASE: [&str; 5] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=3", REMOTE_NODE];
const STORAGE_TIMEOUT_MS: u64 = 2000;
const CMD_TIMEOUT_MS: u64 = 2000;
const SSH_METRICS_TIMEOUT_MS: u64 = 5000;
const LOGS_TIMEOUT_MS: u64 = 5000;
const DROP_CACHES_TIMEOUT_MS: u64 = 10_000;
const STARTING_GRACE_SECS: u64 = 120;

/// One combined ssh probe collecting meminfo / df / GPU apps / docker ps and
/// per-pid details from ai2. Sections are delimited with markers and parsed
/// below. Kept as one ssh round-trip so a slow link cannot stall the loop.
const AI2_METRICS_SCRIPT: &str = r####"echo "###MEM"; awk '/^(MemTotal|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):/{print $1, $2}' /proc/meminfo; echo "###DF"; df -B1 /; echo "###GPU"; nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null; echo "###DOCKER"; docker ps --no-trunc --format "{{.ID}}|{{.Names}}" 2>/dev/null; echo "###PROCS"; for p in $(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null | cut -d, -f1); do echo "PID $p"; grep VmRSS /proc/$p/status 2>/dev/null; grep -o "docker-[0-9a-f]*" /proc/$p/cgroup 2>/dev/null; echo "CWD $(readlink /proc/$p/cwd 2>/dev/null)"; echo "CMDLINE $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"; done"####;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Engine {
    Tensorfold,
    Vllm,
    Velogb10,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Running,
    Stopped,
    Starting,
    Stopping,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Running => "RUNNING",
            Status::Stopped => "STOPPED",
            Status::Starting => "STARTING",
            Status::Stopping => "STOPPING",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entry {
    id: String,
    name: String,
    path: String,
    start_cmd: String,
    stop_cmd: String,
    host: String,
    port: u16,
    slot: String,
    engine: Engine,
    containers: Vec<String>,
    process_match: Option<String>,
}

/// Previous telemetry sample, used to compute deltas.
#[derive(Debug, Clone, Copy)]
struct PrevSample {
    ts: f64,
    completion_tokens: u64,
    decode_seconds: f64,
    gen_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryDto {
    concurrent_requests: Option<u64>,
    queued_requests: Option<u64>,
    live_tok_per_sec: f64,
    lifetime_decode_tok_per_sec: Option<f64>,
    total_tokens_generated: Option<u64>,
    prompt_tokens_total: Option<u64>,
    cached_tokens_total: Option<u64>,
    http_requests_total: Option<u64>,
}

#[derive(Debug, Clone)]
struct EntryRuntime {
    status: Status,
    running_since_unix: Option<u64>,
    prev: Option<PrevSample>,
    telemetry: Option<TelemetryDto>,
    served_model: Option<String>,
    mem_ai_bytes: Option<u64>,
    mem_ai2_bytes: Option<u64>,
}

impl Default for EntryRuntime {
    fn default() -> Self {
        Self {
            status: Status::Stopped,
            running_since_unix: None,
            prev: None,
            telemetry: None,
            served_model: None,
            mem_ai_bytes: None,
            mem_ai2_bytes: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Snapshot types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Default)]
struct DiskStat {
    total_bytes: u64,
    used_bytes: u64,
    avail_bytes: u64,
    use_pct: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct NodeDisk {
    mount: String,
    total_bytes: u64,
    used_bytes: u64,
    avail_bytes: u64,
    use_pct: f64,
}

impl NodeDisk {
    fn new(mount: &str, d: DfLine) -> Self {
        NodeDisk {
            mount: mount.to_string(),
            total_bytes: d.total,
            used_bytes: d.used,
            avail_bytes: d.avail,
            use_pct: d.use_pct,
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
struct NodeInfo {
    mem_total_bytes: u64,
    mem_available_bytes: u64,
    mem_used_bytes: u64,
    mem_cache_bytes: u64,
    swap_total_bytes: u64,
    swap_used_bytes: u64,
    disk: NodeDisk,
}

#[derive(Debug, Clone, Serialize, Default)]
struct StorageChecks {
    ai_mount: bool,
    ai_raid: bool,
    ai_stat: bool,
    ai2_mount: bool,
    ai2_stat: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
struct StorageState {
    online: bool,
    reason: Option<String>,
    checks: StorageChecks,
    disk: DiskStat,
}

#[derive(Debug, Clone, Serialize, Default)]
struct NodesState {
    ai: Option<NodeInfo>,
    ai2: Option<NodeInfo>,
}

#[derive(Debug, Clone, Default)]
struct Snapshot {
    storage: StorageState,
    nodes: NodesState,
}

#[derive(Default)]
struct DashboardState {
    registry: Vec<Entry>,
    runtime: HashMap<String, EntryRuntime>,
    snapshot: Snapshot,
    /// container id or name -> node ("ai" | "ai2"), rebuilt from `docker ps`.
    container_node: HashMap<String, String>,
}

struct AppState {
    state: Mutex<DashboardState>,
    client: reqwest::Client,
}

fn lock_state(st: &Mutex<DashboardState>) -> MutexGuard<'_, DashboardState> {
    st.lock().unwrap_or_else(|p| p.into_inner())
}

// ---------------------------------------------------------------------------
// Time / misc helpers
// ---------------------------------------------------------------------------

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn now_unix_f() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn slugify(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

fn basename(p: &str) -> &str {
    let p = p.trim_end_matches('/');
    p.rsplit('/').next().unwrap_or(p)
}

/// Run a program with args under a hard timeout. kill_on_drop makes sure a
/// timed-out child (e.g. a hung ssh) is killed when the future is dropped.
async fn run_checked(program: &str, args: &[&str], timeout_ms: u64) -> Result<std::process::Output, String> {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match timeout(Duration::from_millis(timeout_ms), cmd.output()).await {
        Ok(Ok(out)) => Ok(out),
        Ok(Err(e)) => Err(format!("failed to spawn {program}: {e}")),
        Err(_) => Err(format!("timed out after {timeout_ms}ms")),
    }
}

async fn run_shell(script: &str, timeout_ms: u64) -> Result<std::process::Output, String> {
    run_checked("bash", &["-c", script], timeout_ms).await
}

async fn ssh_ai2(script: &str, timeout_ms: u64) -> Result<std::process::Output, String> {
    let mut args: Vec<&str> = SSH_BASE.to_vec();
    args.push(script);
    run_checked("ssh", &args, timeout_ms).await
}

async fn fs_read(path: &str) -> Option<String> {
    timeout(Duration::from_secs(CMD_TIMEOUT_MS / 1000), tokio::fs::read_to_string(path))
        .await
        .ok()?
        .ok()
}

async fn fs_read_bytes(path: &str) -> Option<Vec<u8>> {
    timeout(Duration::from_secs(CMD_TIMEOUT_MS / 1000), tokio::fs::read(path))
        .await
        .ok()?
        .ok()
}

// ---------------------------------------------------------------------------
// Storage loop
// ---------------------------------------------------------------------------

struct CheckOut {
    ok: bool,
    reason: Option<String>,
}

fn fail(ok: bool, reason: String) -> CheckOut {
    if ok {
        CheckOut { ok: true, reason: None }
    } else {
        CheckOut { ok: false, reason: Some(reason) }
    }
}

async fn check_mount_ai() -> CheckOut {
    match run_checked("findmnt", &["-n", "-o", "FSTYPE,OPTIONS", "/mnt/nvmeof"], STORAGE_TIMEOUT_MS).await {
        Ok(out) => {
            let has_gfs2 = String::from_utf8_lossy(&out.stdout).contains("gfs2");
            fail(has_gfs2, "ai: findmnt /mnt/nvmeof did not report gfs2".into())
        }
        Err(e) => fail(false, format!("ai: findmnt /mnt/nvmeof {e}")),
    }
}

async fn check_raid_ai() -> CheckOut {
    match fs_read("/proc/mdstat").await {
        Some(text) => {
            let ok = text.contains("md0 : active raid0");
            fail(ok, "ai: /proc/mdstat does not show 'md0 : active raid0'".into())
        }
        None => fail(false, "ai: failed to read /proc/mdstat".into()),
    }
}

async fn check_stat_ai() -> CheckOut {
    match run_checked("stat", &["-t", "/mnt/nvmeof/repos", "/mnt/nvmeof/huggingface"], STORAGE_TIMEOUT_MS).await {
        Ok(out) => fail(out.status.success(), "ai: stat /mnt/nvmeof/repos /mnt/nvmeof/huggingface failed".into()),
        Err(e) => fail(false, format!("ai: stat /mnt/nvmeof {e}")),
    }
}

/// Single ssh for ai2: findmnt output governs ai2_mount, the (stat-governed)
/// process exit code governs ai2_stat.
async fn check_ai2() -> (CheckOut, CheckOut) {
    let script = "findmnt -n -o FSTYPE,OPTIONS /mnt/nvmeof; stat -t /mnt/nvmeof/repos /mnt/nvmeof/huggingface";
    match ssh_ai2(script, STORAGE_TIMEOUT_MS).await {
        Ok(out) => {
            let has_gfs2 = String::from_utf8_lossy(&out.stdout).contains("gfs2");
            let mount = fail(has_gfs2, "ai2: findmnt /mnt/nvmeof did not report gfs2".into());
            let code = out.status.code().unwrap_or(-1);
            let stat = fail(
                out.status.success(),
                format!("ai2: stat /mnt/nvmeof/repos /mnt/nvmeof/huggingface failed (exit code {code})"),
            );
            (mount, stat)
        }
        Err(e) => {
            let msg = if e.contains("timed out") {
                format!("ai2: stat /mnt/nvmeof {e}")
            } else {
                format!("ai2: ssh {e}")
            };
            (fail(false, msg.clone()), fail(false, msg))
        }
    }
}

async fn storage_cycle(state: Arc<AppState>) -> Result<(), String> {
    let c0 = check_mount_ai().await;
    let c1 = check_raid_ai().await;
    let c2 = check_stat_ai().await;
    let (c3, c4) = check_ai2().await;

    let reasons: Vec<String> = [&c0, &c1, &c2, &c3, &c4]
        .iter()
        .filter_map(|c| c.reason.clone())
        .collect();
    let online = [&c0, &c1, &c2, &c3, &c4].iter().all(|c| c.ok);
    let reason = reasons.into_iter().next();

    let mut st = lock_state(&state.state);
    st.snapshot.storage.checks.ai_mount = c0.ok;
    st.snapshot.storage.checks.ai_raid = c1.ok;
    st.snapshot.storage.checks.ai_stat = c2.ok;
    st.snapshot.storage.checks.ai2_mount = c3.ok;
    st.snapshot.storage.checks.ai2_stat = c4.ok;
    st.snapshot.storage.online = online;
    st.snapshot.storage.reason = reason;
    Ok(())
}

async fn storage_loop(state: Arc<AppState>) {
    loop {
        match tokio::spawn(storage_cycle(state.clone())).await {
            Ok(Err(e)) => eprintln!("[storage] cycle error: {e}"),
            Err(e) => eprintln!("[storage] cycle panicked: {e}"),
            _ => {}
        }
        sleep(Duration::from_secs(3)).await;
    }
}

// ---------------------------------------------------------------------------
// Metrics collection
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct GpuProc {
    pid: u64,
    mem_mib: u64,
    rss: Option<u64>,
    container: Option<String>,
    cwd: Option<String>,
    cmdline: String,
}

#[derive(Debug, Default)]
struct NodeSample {
    info: NodeInfo,
    /// df of /mnt/nvmeof on ai (None on ai2 or if df failed).
    storage_df: Option<DfLine>,
    procs: Vec<GpuProc>,
    /// container id -> name, from `docker ps`.
    docker: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct DfLine {
    total: u64,
    used: u64,
    avail: u64,
    use_pct: f64,
}

fn parse_meminfo(text: &str) -> HashMap<String, u64> {
    let mut m = HashMap::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if let Some(val) = v.trim().split_whitespace().next().and_then(|t| t.parse::<u64>().ok()) {
                m.insert(k.trim().to_string(), val);
            }
        }
    }
    m
}

/// Parse `df -B1 <path>` output. Handles the long-device-name wrap case
/// (5 fields followed by a lone mount token on the next line).
fn parse_df(out: &str, mount: &str) -> Option<DfLine> {
    let mut pending: Option<DfLine> = None;
    let mut first_valid: Option<DfLine> = None;
    for line in out.lines() {
        if line.starts_with("Filesystem") {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        if let Some(p) = pending.take() {
            if t[0] == mount {
                return Some(p);
            }
            if first_valid.is_none() {
                first_valid = Some(p);
            }
            continue;
        }
        if t.len() >= 6 {
            let d = DfLine {
                total: t[1].parse().unwrap_or(0),
                used: t[2].parse().unwrap_or(0),
                avail: t[3].parse().unwrap_or(0),
                use_pct: t[4].trim_end_matches('%').parse::<f64>().unwrap_or(0.0),
            };
            if t[5] == mount {
                return Some(d);
            }
            if first_valid.is_none() {
                first_valid = Some(d);
            }
        } else if t.len() == 5 {
            pending = Some(DfLine {
                total: t[1].parse().unwrap_or(0),
                used: t[2].parse().unwrap_or(0),
                avail: t[3].parse().unwrap_or(0),
                use_pct: t[4].trim_end_matches('%').parse::<f64>().unwrap_or(0.0),
            });
        }
    }
    first_valid
}

/// Parse `nvidia-smi --query-compute-apps=...` CSV lines. Handles both
/// `pid, process_name, mem` and `pid, mem` shapes (mem is always the last field).
fn parse_gpu_apps(out: &str) -> Vec<(u64, String, u64)> {
    out.lines()
        .filter_map(|l| {
            let parts: Vec<&str> = l.split(',').map(str::trim).collect();
            if parts.len() < 2 {
                return None;
            }
            let pid = parts[0].parse::<u64>().ok()?;
            let mem = parts[parts.len() - 1].parse::<u64>().ok()?;
            let name = if parts.len() >= 3 { parts[1].to_string() } else { String::new() };
            Some((pid, name, mem))
        })
        .collect()
}

/// Parse `docker ps --format {{.ID}}|{{.Names}}` (with tab fallbacks).
fn parse_docker_ps(text: &str) -> HashMap<String, String> {
    let mut m = HashMap::new();
    for line in text.lines() {
        let line = line.replace('\t', "|").replace("\\t", "|");
        if let Some((id, name)) = line.split_once('|') {
            let id = id.trim();
            if !id.is_empty() {
                m.insert(id.to_string(), name.trim().to_string());
            }
        } else {
            let mut it = line.split_whitespace();
            if let Some(id) = it.next() {
                let name = it.collect::<Vec<_>>().join(" ");
                m.insert(id.to_string(), name);
            }
        }
    }
    m
}

async fn enrich_pid_local(pid: u64, mem_mib: u64) -> GpuProc {
    let mut p = GpuProc {
        pid,
        mem_mib,
        ..Default::default()
    };
    if let Some(text) = fs_read(&format!("/proc/{pid}/status")).await {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.trim().split_whitespace().next().and_then(|t| t.parse::<u64>().ok()) {
                    p.rss = Some(kb * 1024);
                }
            }
        }
    }
    if let Some(text) = fs_read(&format!("/proc/{pid}/cgroup")).await {
        for line in text.lines() {
            if let Some(idx) = line.find("docker-") {
                let id = line[idx + 7..].trim_end_matches(".scope");
                if !id.is_empty() {
                    p.container = Some(id.to_string());
                }
            }
        }
    }
    if let Ok(Ok(link)) =
        timeout(Duration::from_secs(CMD_TIMEOUT_MS / 1000), tokio::fs::read_link(format!("/proc/{pid}/cwd"))).await
    {
        p.cwd = Some(link.to_string_lossy().to_string());
    }
    if let Some(bytes) = fs_read_bytes(&format!("/proc/{pid}/cmdline")).await {
        p.cmdline = bytes
            .split(|&b| b == 0)
            .filter(|s| !s.is_empty())
            .map(|b| String::from_utf8_lossy(b).to_string())
            .collect::<Vec<_>>()
            .join(" ");
    }
    p
}

fn build_node_info(m: &HashMap<String, u64>, disk: NodeDisk) -> NodeInfo {
    let kb = |k: &str| m.get(k).copied().unwrap_or(0) * 1024;
    let total = kb("MemTotal");
    let avail = kb("MemAvailable");
    let swap_total = kb("SwapTotal");
    let swap_free = kb("SwapFree");
    NodeInfo {
        mem_total_bytes: total,
        mem_available_bytes: avail,
        mem_used_bytes: total.saturating_sub(avail),
        mem_cache_bytes: kb("Buffers") + kb("Cached"),
        swap_total_bytes: swap_total,
        swap_used_bytes: swap_total.saturating_sub(swap_free),
        disk,
    }
}

async fn collect_local() -> NodeSample {
    let mut sample = NodeSample::default();

    if let Some(text) = fs_read("/proc/meminfo").await {
        let m = parse_meminfo(&text);
        sample.info = build_node_info(&m, NodeDisk { mount: "/".into(), ..Default::default() });
    }

    if let Ok(out) = run_checked("df", &["-B1", "/"], CMD_TIMEOUT_MS).await {
        if let Some(d) = parse_df(&String::from_utf8_lossy(&out.stdout), "/") {
            sample.info.disk = NodeDisk::new("/", d);
        }
    }

    if let Ok(out) = run_checked("df", &["-B1", "/mnt/nvmeof"], CMD_TIMEOUT_MS).await {
        sample.storage_df = parse_df(&String::from_utf8_lossy(&out.stdout), "/mnt/nvmeof");
    }

    if let Ok(out) = run_checked(
        "nvidia-smi",
        &["--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        CMD_TIMEOUT_MS,
    )
    .await
    {
        for (pid, _name, mem) in parse_gpu_apps(&String::from_utf8_lossy(&out.stdout)) {
            sample.procs.push(enrich_pid_local(pid, mem).await);
        }
    }

    if let Ok(out) = run_checked("docker", &["ps", "--no-trunc", "--format", "{{.ID}}|{{.Names}}"], CMD_TIMEOUT_MS).await {
        sample.docker = parse_docker_ps(&String::from_utf8_lossy(&out.stdout));
    }

    sample
}

fn parse_ai2_sample(text: &str) -> NodeSample {
    let mut section = "";
    let mut mem_raw: Vec<&str> = Vec::new();
    let mut df_raw: Vec<&str> = Vec::new();
    let mut gpu_raw: Vec<&str> = Vec::new();
    let mut docker_raw: Vec<&str> = Vec::new();
    let mut procs_raw: Vec<&str> = Vec::new();
    for line in text.lines() {
        match line.trim() {
            "###MEM" => {
                section = "mem";
                continue;
            }
            "###DF" => {
                section = "df";
                continue;
            }
            "###GPU" => {
                section = "gpu";
                continue;
            }
            "###DOCKER" => {
                section = "docker";
                continue;
            }
            "###PROCS" => {
                section = "procs";
                continue;
            }
            _ => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        match section {
            "mem" => mem_raw.push(line),
            "df" => df_raw.push(line),
            "gpu" => gpu_raw.push(line),
            "docker" => docker_raw.push(line),
            "procs" => procs_raw.push(line),
            _ => {}
        }
    }

    let mut sample = NodeSample::default();
    let m = parse_meminfo(&mem_raw.join("\n"));
    let disk = parse_df(&df_raw.join("\n"), "/")
        .map(|d| NodeDisk::new("/", d))
        .unwrap_or_else(|| NodeDisk { mount: "/".into(), ..Default::default() });
    sample.info = build_node_info(&m, disk);

    let gpu_mem: HashMap<u64, u64> = parse_gpu_apps(&gpu_raw.join("\n")).into_iter().map(|(p, _, mi)| (p, mi)).collect();
    sample.docker = parse_docker_ps(&docker_raw.join("\n"));

    let mut cur: Option<GpuProc> = None;
    for line in procs_raw {
        if let Some(rest) = line.strip_prefix("PID ") {
            if let Some(c) = cur.take() {
                sample.procs.push(c);
            }
            let pid: u64 = rest.trim().parse().unwrap_or(0);
            let mut np = GpuProc::default();
            np.pid = pid;
            np.mem_mib = gpu_mem.get(&pid).copied().unwrap_or(0);
            cur = Some(np);
            continue;
        }
        if let Some(c) = cur.as_mut() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Some(kb) = rest.trim().split_whitespace().next().and_then(|t| t.parse::<u64>().ok()) {
                    c.rss = Some(kb * 1024);
                }
            } else if line.starts_with("docker-") {
                let id = line.trim_start_matches("docker-").trim_end_matches(".scope");
                if !id.is_empty() {
                    c.container = Some(id.to_string());
                }
            } else if let Some(rest) = line.strip_prefix("CWD ") {
                if !rest.is_empty() {
                    c.cwd = Some(rest.to_string());
                }
            } else if let Some(rest) = line.strip_prefix("CMDLINE ") {
                c.cmdline = rest.to_string();
            }
        }
    }
    if let Some(c) = cur.take() {
        sample.procs.push(c);
    }
    sample
}

async fn collect_ai2() -> Option<NodeSample> {
    let out = ssh_ai2(AI2_METRICS_SCRIPT, SSH_METRICS_TIMEOUT_MS).await.ok()?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    Some(parse_ai2_sample(&text))
}

/// Map a GPU process to a registry entry id.
/// Order: container id/name match -> cwd == path -> process_match -> cmdline
/// contains basename(path).
fn entry_for_proc(p: &GpuProc, node_docker: &HashMap<String, String>, registry: &[Entry]) -> Option<String> {
    if let Some(cid) = &p.container {
        let cname = node_docker.get(cid.as_str());
        for e in registry {
            for ec in &e.containers {
                if cid.starts_with(ec.as_str()) || ec.starts_with(cid.as_str()) {
                    return Some(e.id.clone());
                }
                if let Some(n) = cname {
                    if n == ec {
                        return Some(e.id.clone());
                    }
                }
            }
        }
    }
    for e in registry {
        if let Some(cwd) = &p.cwd {
            if !cwd.is_empty() && cwd == &e.path {
                return Some(e.id.clone());
            }
        }
    }
    for e in registry {
        if let Some(pm) = &e.process_match {
            if !pm.is_empty() && p.cmdline.contains(pm.as_str()) {
                return Some(e.id.clone());
            }
        }
    }
    for e in registry {
        let b = basename(&e.path);
        if !b.is_empty() && p.cmdline.contains(b) {
            return Some(e.id.clone());
        }
    }
    None
}

async fn metrics_cycle(state: Arc<AppState>) -> Result<(), String> {
    let registry: Vec<Entry> = {
        let st = lock_state(&state.state);
        st.registry.clone()
    };

    // Collect both nodes. Individual commands carry their own 2s timeouts; the
    // whole local section is capped at 5s.
    let ai = match timeout(Duration::from_secs(5), collect_local()).await {
        Ok(s) => Some(s),
        Err(_) => None, // keep last good
    };
    let ai2 = collect_ai2().await; // None => keep last good

    // Attribute GPU+RSS memory per entry per node; detect process_match.
    // Index 0 = ai, 1 = ai2.
    let mut mem: HashMap<String, [Option<u64>; 2]> = HashMap::new();
    for (idx, sample) in [(0usize, &ai), (1usize, &ai2)] {
        if let Some(s) = sample {
            for p in &s.procs {
                if let Some(id) = entry_for_proc(p, &s.docker, &registry) {
                    let bytes = p.mem_mib * 1024 * 1024 + p.rss.unwrap_or(0);
                    let slot = mem.entry(id).or_insert([None, None]);
                    slot[idx] = Some(slot[idx].unwrap_or(0) + bytes);
                }
            }
        }
    }
    let mut proc_seen: HashSet<String> = HashSet::new();
    for sample in [&ai, &ai2].into_iter().flatten() {
        for p in &sample.procs {
            for e in &registry {
                if let Some(pm) = &e.process_match {
                    if !pm.is_empty() && p.cmdline.contains(pm.as_str()) {
                        proc_seen.insert(e.id.clone());
                    }
                }
            }
        }
    }

    let now = now_unix();
    {
        let mut st = lock_state(&state.state);

        // Rebuild container -> node map per successful collection.
        if let Some(s) = &ai {
            st.container_node.retain(|_, v| v != LOCAL_NODE);
            for (id, name) in &s.docker {
                st.container_node.insert(id.clone(), LOCAL_NODE.to_string());
                if !name.is_empty() {
                    st.container_node.insert(name.clone(), LOCAL_NODE.to_string());
                }
            }
        }
        if let Some(s) = &ai2 {
            st.container_node.retain(|_, v| v != REMOTE_NODE);
            for (id, name) in &s.docker {
                st.container_node.insert(id.clone(), REMOTE_NODE.to_string());
                if !name.is_empty() {
                    st.container_node.insert(name.clone(), REMOTE_NODE.to_string());
                }
            }
        }

        // Snapshot updates (stale nodes keep last good values).
        if let Some(s) = &ai {
            st.snapshot.nodes.ai = Some(s.info.clone());
            if let Some(d) = s.storage_df {
                st.snapshot.storage.disk = DiskStat {
                    total_bytes: d.total,
                    used_bytes: d.used,
                    avail_bytes: d.avail,
                    use_pct: d.use_pct,
                };
            }
        }
        if let Some(s) = &ai2 {
            st.snapshot.nodes.ai2 = Some(s.info.clone());
        }

        // Status transitions + memory attribution.
        for e in &registry {
            let containers_running = e.containers.iter().any(|c| st.container_node.contains_key(c.as_str()));
            let infra = containers_running || proc_seen.contains(&e.id);
            if let Some(rt) = st.runtime.get_mut(&e.id) {
                match rt.status {
                    Status::Stopped | Status::Starting => {
                        if infra {
                            if rt.running_since_unix.is_none() {
                                rt.running_since_unix = Some(now);
                            }
                            rt.status = Status::Running;
                        }
                    }
                    Status::Stopping => {
                        if !infra {
                            rt.status = Status::Stopped;
                            rt.running_since_unix = None;
                        }
                    }
                    Status::Running => {}
                }
                if rt.status == Status::Starting {
                    if let Some(since) = rt.running_since_unix {
                        if now.saturating_sub(since) > STARTING_GRACE_SECS && !infra {
                            rt.status = Status::Stopped;
                            rt.running_since_unix = None;
                        }
                    }
                }
                let m = mem.get(&e.id);
                rt.mem_ai_bytes = m.and_then(|x| x[0]);
                rt.mem_ai2_bytes = m.and_then(|x| x[1]);
            }
        }
    }
    Ok(())
}

async fn metrics_loop(state: Arc<AppState>) {
    loop {
        match tokio::spawn(metrics_cycle(state.clone())).await {
            Ok(Err(e)) => eprintln!("[metrics] cycle error: {e}"),
            Err(e) => eprintln!("[metrics] cycle panicked: {e}"),
            _ => {}
        }
        sleep(Duration::from_secs(3)).await;
    }
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

enum RawTel {
    Tensorfold {
        concurrent: Option<u64>,
        completion: u64,
        decode: f64,
        prompt: Option<u64>,
        cached: Option<u64>,
    },
    Vllm {
        concurrent: Option<u64>,
        queued: Option<u64>,
        gen_total: Option<u64>,
        http_total: Option<u64>,
        served: Option<String>,
    },
    Velogb10 {
        total: u64,
        step_ms: Option<f64>,
        yield_per_step: Option<f64>,
    },
}

async fn http_json(client: &reqwest::Client, url: &str) -> Result<Value, String> {
    let resp = client.get(url).send().await.map_err(|e| format!("GET {url} failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("GET {url} returned {}", resp.status()));
    }
    resp.json::<Value>().await.map_err(|e| format!("GET {url} returned invalid json: {e}"))
}

fn prom_value(line: &str) -> Option<u64> {
    let f: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    if f < 0.0 {
        return None;
    }
    Some(f as u64)
}

async fn fetch_tensorfold(client: &reqwest::Client, e: &Entry) -> Result<RawTel, String> {
    let v = http_json(client, &format!("http://{}:{}/health", e.host, e.port)).await?;
    let concurrent = v
        .get("requests_running")
        .and_then(Value::as_u64)
        .or_else(|| v.get("inflight").and_then(Value::as_u64));
    let completion = v.get("completion_tokens_total").and_then(Value::as_f64).unwrap_or(0.0) as u64;
    let decode = v.get("decode_seconds_total").and_then(Value::as_f64).unwrap_or(0.0);
    let prompt = v.get("prompt_tokens_total").and_then(Value::as_u64);
    let cached = v.get("cached_tokens_total").and_then(Value::as_u64);
    Ok(RawTel::Tensorfold { concurrent, completion, decode, prompt, cached })
}

async fn fetch_vllm(client: &reqwest::Client, e: &Entry) -> Result<RawTel, String> {
    let models = http_json(client, &format!("http://{}:{}/v1/models", e.host, e.port)).await?;
    let served = models
        .get("data")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(Value::as_str)
        .map(|s| s.to_string());

    let mut concurrent: Option<u64> = None;
    let mut queued: Option<u64> = None;
    let mut gen_total: Option<u64> = None;
    let mut http_total: Option<u64> = None;
    if let Ok(resp) = client.get(format!("http://{}:{}/metrics", e.host, e.port)).send().await {
        if let Ok(text) = resp.text().await {
            for line in text.lines() {
                let metric = line.split_whitespace().next().unwrap_or("");
                match metric {
                    "vllm:num_requests_running" => concurrent = prom_value(line).or(concurrent),
                    "vllm:num_requests_waiting" => queued = prom_value(line).or(queued),
                    "vllm:generation_tokens_total" => gen_total = prom_value(line).or(gen_total),
                    m if m == "http_requests_total" || m.starts_with("http_requests_total{") => {
                        if let Some(v) = prom_value(line) {
                            *http_total.get_or_insert(0) += v;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(RawTel::Vllm { concurrent, queued, gen_total, http_total, served })
}

async fn fetch_velogb10(client: &reqwest::Client, e: &Entry) -> Result<RawTel, String> {
    let v = http_json(client, &format!("http://{}:{}/health", e.host, e.port)).await?;
    let mtp = v.get("telemetry").and_then(|t| t.get("mtp"));
    let dflash = v.get("telemetry").and_then(|t| t.get("dflash2"));
    let mtp_emitted = mtp.and_then(|m| m.get("emitted")).and_then(Value::as_u64).unwrap_or(0);
    let df_emitted = dflash.and_then(|m| m.get("emitted")).and_then(Value::as_u64).unwrap_or(0);
    let step_ms = v.get("telemetry").and_then(|t| t.get("step_ms_p50")).and_then(Value::as_f64);
    let yield_per_step = mtp.and_then(|m| m.get("yield_tok_per_step")).and_then(Value::as_f64);
    Ok(RawTel::Velogb10 { total: mtp_emitted + df_emitted, step_ms, yield_per_step })
}

/// Compute the telemetry DTO from a raw sample using (and updating) the
/// previous sample stored on the runtime.
fn compute_dto(raw: RawTel, ts: f64, rt: &mut EntryRuntime) -> TelemetryDto {
    let prev = rt.prev.take();
    let (dto, prev) = match raw {
        RawTel::Tensorfold { concurrent, completion, decode, prompt, cached } => {
            rt.served_model = None;
            let live = match &prev {
                Some(p) => {
                    let dd = decode - p.decode_seconds;
                    if dd > f64::EPSILON {
                        (completion as f64 - p.completion_tokens as f64) / dd
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
            let dto = TelemetryDto {
                concurrent_requests: concurrent,
                queued_requests: None,
                live_tok_per_sec: live.max(0.0),
                lifetime_decode_tok_per_sec: if decode > 0.0 { Some(completion as f64 / decode) } else { None },
                total_tokens_generated: Some(completion),
                prompt_tokens_total: prompt,
                cached_tokens_total: cached,
                http_requests_total: None,
            };
            (dto, PrevSample { ts, completion_tokens: completion, decode_seconds: decode, gen_tokens: 0 })
        }
        RawTel::Vllm { concurrent, queued, gen_total, http_total, served } => {
            rt.served_model = served;
            let live = match (&prev, &gen_total) {
                (Some(p), Some(g)) => {
                    let dt = ts - p.ts;
                    if dt > 0.0 && *g >= p.gen_tokens {
                        (*g - p.gen_tokens) as f64 / dt
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            let dto = TelemetryDto {
                concurrent_requests: concurrent,
                queued_requests: queued,
                live_tok_per_sec: live.max(0.0),
                lifetime_decode_tok_per_sec: None,
                total_tokens_generated: gen_total,
                prompt_tokens_total: None,
                cached_tokens_total: None,
                http_requests_total: http_total,
            };
            (dto, PrevSample { ts, completion_tokens: 0, decode_seconds: 0.0, gen_tokens: gen_total.unwrap_or(0) })
        }
        RawTel::Velogb10 { total, step_ms, yield_per_step } => {
            rt.served_model = None;
            let mut live = 0.0;
            if let Some(p) = &prev {
                let dt = ts - p.ts;
                if dt > 0.0 && total > p.gen_tokens {
                    live = (total - p.gen_tokens) as f64 / dt;
                }
            }
            if live <= 0.0 {
                if let (Some(ms), Some(y)) = (step_ms, yield_per_step) {
                    if ms > 0.0 && y > 0.0 {
                        live = y * 1000.0 / ms;
                    }
                }
            }
            let dto = TelemetryDto {
                concurrent_requests: None,
                queued_requests: None,
                live_tok_per_sec: live.max(0.0),
                lifetime_decode_tok_per_sec: None,
                total_tokens_generated: Some(total),
                prompt_tokens_total: None,
                cached_tokens_total: None,
                http_requests_total: None,
            };
            (dto, PrevSample { ts, completion_tokens: 0, decode_seconds: 0.0, gen_tokens: total })
        }
    };
    rt.prev = Some(prev);
    dto
}

async fn telemetry_cycle(state: Arc<AppState>) -> Result<(), String> {
    let targets: Vec<Entry> = {
        let st = lock_state(&state.state);
        st.registry
            .iter()
            .filter(|e| {
                matches!(
                    st.runtime.get(&e.id).map(|r| r.status),
                    Some(Status::Running) | Some(Status::Starting)
                )
            })
            .cloned()
            .collect()
    };

    for e in targets {
        let raw = match e.engine {
            Engine::Tensorfold => fetch_tensorfold(&state.client, &e).await,
            Engine::Vllm => fetch_vllm(&state.client, &e).await,
            Engine::Velogb10 => fetch_velogb10(&state.client, &e).await,
        };
        let ts = now_unix_f();
        let now = now_unix();
        let mut st = lock_state(&state.state);
        let infra = st.container_node.keys().any(|c| e.containers.iter().any(|ec| ec == c));
        let Some(rt) = st.runtime.get_mut(&e.id) else { continue };
        match raw {
            Ok(raw) => {
                let dto = compute_dto(raw, ts, rt);
                if rt.status != Status::Running {
                    if rt.running_since_unix.is_none() {
                        rt.running_since_unix = Some(now);
                    }
                    rt.status = Status::Running;
                }
                rt.telemetry = Some(dto);
            }
            Err(_) => {
                rt.telemetry = None;
                rt.prev = None;
                if !infra && matches!(rt.status, Status::Running | Status::Starting) {
                    rt.status = Status::Stopped;
                    rt.running_since_unix = None;
                }
            }
        }
    }
    Ok(())
}

async fn telemetry_loop(state: Arc<AppState>) {
    loop {
        match tokio::spawn(telemetry_cycle(state.clone())).await {
            Ok(Err(e)) => eprintln!("[telemetry] cycle error: {e}"),
            Err(e) => eprintln!("[telemetry] cycle panicked: {e}"),
            _ => {}
        }
        sleep(Duration::from_secs(2)).await;
    }
}

// ---------------------------------------------------------------------------
// Registry persistence & seed
// ---------------------------------------------------------------------------

async fn persist_registry(registry: &[Entry]) {
    match serde_json::to_string_pretty(registry) {
        Ok(text) => {
            let tmp = "data/registry.json.tmp";
            if tokio::fs::write(tmp, text).await.is_ok() {
                let _ = tokio::fs::rename(tmp, "data/registry.json").await;
            }
        }
        Err(e) => eprintln!("[registry] serialize failed: {e}"),
    }
}

fn seed_registry() -> Vec<Entry> {
    vec![
        Entry {
            id: "glm53-tensorfold-spark".into(),
            name: "glm53-tensorfold-spark".into(),
            path: "/home/sean/repos/glm53-tensorfold-spark".into(),
            start_cmd: "./launch.sh start".into(),
            stop_cmd: "./launch.sh stop".into(),
            host: "192.168.20.63".into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: Engine::Tensorfold,
            containers: vec!["glm53-tf-r0".into(), "glm53-tf-r1".into()],
            process_match: None,
        },
        Entry {
            id: "deepseek-v4.1-flash-2x-dgx-spark".into(),
            name: "DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            path: "/home/sean/repos/DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            start_cmd: "./start-server.sh".into(),
            stop_cmd: "./stop-server.sh".into(),
            host: "192.168.20.63".into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: Engine::Vllm,
            containers: vec![],
            process_match: None,
        },
        Entry {
            id: "velogb10-qwen38-flash-next".into(),
            name: "veloGB10 (Qwen3.8-Flash-Next)".into(),
            path: "/home/sean/repos/veloGB10".into(),
            start_cmd: "./launch_qwen38_flash_next.sh 2".into(),
            stop_cmd: "pkill -f gb10_inference || true; ssh -o BatchMode=yes ai2 'pkill -f gb10_inference || true'".into(),
            host: "192.168.20.63".into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: Engine::Velogb10,
            containers: vec![],
            process_match: Some("gb10_inference".into()),
        },
        Entry {
            id: "launch-embedding-ai-bge-m3".into(),
            name: "launch-embedding-ai.sh (BGE-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start_cmd: "./launch-embedding-ai.sh".into(),
            stop_cmd: "docker rm -f vllm_bge_m3".into(),
            host: "192.168.20.63".into(),
            port: 8891,
            slot: "aux-embedding".into(),
            engine: Engine::Vllm,
            containers: vec!["vllm_bge_m3".into()],
            process_match: None,
        },
        Entry {
            id: "launch-reranker-ai2-bge-reranker".into(),
            name: "launch-reranker-ai2.sh (BGE-Reranker-v2-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start_cmd: "./launch-reranker-ai2.sh".into(),
            stop_cmd: "ssh -o BatchMode=yes ai2 'docker rm -f vllm_bge_reranker'".into(),
            host: "192.168.20.229".into(),
            port: 8892,
            slot: "aux-reranker".into(),
            engine: Engine::Vllm,
            containers: vec!["vllm_bge_reranker".into()],
            process_match: None,
        },
    ]
}

/// Load the registry; seeds the default 5 entries when missing/invalid.
/// Returns (registry, seeded).
async fn load_registry() -> (Vec<Entry>, bool) {
    if let Ok(text) = tokio::fs::read_to_string("data/registry.json").await {
        if let Ok(v) = serde_json::from_str::<Vec<Entry>>(&text) {
            if !v.is_empty() {
                return (v, false);
            }
        }
    }
    (seed_registry(), true)
}

// ---------------------------------------------------------------------------
// DTO builders
// ---------------------------------------------------------------------------

fn entry_dto(e: &Entry, rt: Option<&EntryRuntime>, now: u64) -> Value {
    let (status, uptime) = match rt {
        Some(r) => (
            r.status.as_str(),
            r.running_since_unix.map(|s| now.saturating_sub(s)).unwrap_or(0),
        ),
        None => ("STOPPED", 0),
    };
    let (mem_ai, mem_ai2) = rt.map(|r| (r.mem_ai_bytes, r.mem_ai2_bytes)).unwrap_or((None, None));
    let mem_total = match (mem_ai, mem_ai2) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    json!({
        "id": e.id,
        "name": e.name,
        "path": e.path,
        "start_cmd": e.start_cmd,
        "stop_cmd": e.stop_cmd,
        "host": e.host,
        "port": e.port,
        "slot": e.slot,
        "engine": e.engine,
        "containers": e.containers,
        "process_match": e.process_match,
        "status": status,
        "uptime_s": uptime,
        "served_model": rt.and_then(|r| r.served_model.clone()),
        "mem_ai_bytes": mem_ai,
        "mem_ai2_bytes": mem_ai2,
        "mem_total_bytes": mem_total,
        "telemetry": rt.and_then(|r| r.telemetry.clone()),
    })
}

// ---------------------------------------------------------------------------
// Process spawning with log capture
// ---------------------------------------------------------------------------

async fn append_log_line(path: &str, line: &str) {
    if let Ok(mut f) = tokio::fs::OpenOptions::new().create(true).append(true).open(path).await {
        let _ = f.write_all(format!("{line}\n").as_bytes()).await;
    }
}

async fn pipe_to_log<R: AsyncRead + Unpin + Send + 'static>(mut pipe: R, path: String) {
    let Ok(mut file) = tokio::fs::OpenOptions::new().create(true).append(true).open(&path).await else {
        return;
    };
    let mut buf = [0u8; 4096];
    loop {
        match pipe.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let _ = file.write_all(&buf[..n]).await;
            }
        }
    }
}

/// Spawn `bash -c <cmd>` in the entry's directory, appending stdout+stderr to
/// logs/<id>.log. Never awaited to completion; a waiter task reaps the child.
async fn spawn_logged_cmd(entry: &Entry, cmd: &str) -> Result<(), String> {
    let log_path = format!("logs/{}.log", entry.id);
    let _ = tokio::fs::create_dir_all("logs").await;

    let cwd_ok = tokio::fs::metadata(&entry.path).await.map(|m| m.is_dir()).unwrap_or(false);
    let mut builder = Command::new("bash");
    builder.arg("-c").arg(cmd);
    if cwd_ok {
        builder.current_dir(&entry.path);
    }
    builder.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = builder.spawn().map_err(|e| format!("failed to spawn command: {e}"))?;

    if let Some(out) = child.stdout.take() {
        tokio::spawn(pipe_to_log(out, log_path.clone()));
    }
    if let Some(err) = child.stderr.take() {
        tokio::spawn(pipe_to_log(err, log_path.clone()));
    }
    tokio::spawn(async move {
        let mut child = child;
        let _ = child.wait().await;
        append_log_line(&log_path, "[process exited]").await;
    });
    Ok(())
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

fn bad_request(msg: &str) -> (StatusCode, Json<Value>) {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": msg })))
}

async fn root() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn api_state(State(state): State<Arc<AppState>>) -> Json<Value> {
    let (snap, reg, rt_map) = {
        let st = lock_state(&state.state);
        (st.snapshot.clone(), st.registry.clone(), st.runtime.clone())
    };
    let now = now_unix();
    let entries: Vec<Value> = reg.iter().map(|e| entry_dto(e, rt_map.get(&e.id), now)).collect();
    Json(json!({
        "ts": now,
        "storage": snap.storage,
        "nodes": snap.nodes,
        "entries": entries,
    }))
}

#[derive(Deserialize)]
struct EntryInput {
    name: String,
    path: String,
    #[serde(default)]
    start_cmd: String,
    #[serde(default)]
    stop_cmd: String,
    #[serde(default = "default_host")]
    host: String,
    #[serde(default = "default_port")]
    port: u16,
    #[serde(default)]
    slot: String,
    #[serde(default = "default_engine")]
    engine: String,
    #[serde(default)]
    containers: Vec<String>,
    #[serde(default)]
    process_match: Option<String>,
}

fn default_host() -> String {
    "192.168.20.63".into()
}
fn default_port() -> u16 {
    8888
}
fn default_engine() -> String {
    "vllm".into()
}

async fn add_entry(
    State(state): State<Arc<AppState>>,
    Json(input): Json<EntryInput>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if input.name.trim().is_empty() || input.path.trim().is_empty() {
        return Err(bad_request("name and path are required"));
    }
    let id = slugify(&input.name);
    if id.is_empty() {
        return Err(bad_request("could not derive an id from name"));
    }
    let engine = match input.engine.as_str() {
        "tensorfold" => Engine::Tensorfold,
        "velogb10" => Engine::Velogb10,
        _ => Engine::Vllm,
    };
    let entry = Entry {
        id,
        name: input.name,
        path: input.path,
        start_cmd: input.start_cmd,
        stop_cmd: input.stop_cmd,
        host: input.host,
        port: input.port,
        slot: input.slot,
        engine,
        containers: input.containers,
        process_match: input.process_match,
    };

    let duplicate = {
        let st = lock_state(&state.state);
        st.registry.iter().any(|e| e.id == entry.id)
    };
    if duplicate {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({ "error": format!("entry '{}' already exists", entry.id) })),
        ));
    }
    let dto = entry_dto(&entry, None, now_unix());
    let reg = {
        let mut st = lock_state(&state.state);
        st.registry.push(entry.clone());
        st.runtime.insert(entry.id.clone(), EntryRuntime::default());
        st.registry.clone()
    };
    persist_registry(&reg).await;
    Ok(Json(dto))
}

async fn inspect_entry(Json(body): Json<Value>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let Some(path) = body.get("path").and_then(Value::as_str) else {
        return Ok(Json(json!({ "ok": false, "error": "path not found" })));
    };
    match tokio::fs::metadata(path).await {
        Ok(m) if m.is_dir() => {}
        _ => return Ok(Json(json!({ "ok": false, "error": "path not found" }))),
    }

    let mut files: Vec<String> = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir(path).await {
        while let Ok(Some(ent)) = rd.next_entry().await {
            if let Ok(ft) = ent.file_type().await {
                if ft.is_file() {
                    if let Some(name) = ent.file_name().to_str() {
                        files.push(name.to_string());
                    }
                }
            }
        }
    }
    files.sort();

    let name = basename(path).to_string();

    // start/stop script detection.
    let mut start_cmd = String::new();
    let mut stop_cmd = String::new();
    if files.iter().any(|f| f == "launch.sh") {
        start_cmd = "./launch.sh start".into();
        stop_cmd = "./launch.sh stop".into();
    } else if files.iter().any(|f| f == "start-server.sh") {
        start_cmd = "./start-server.sh".into();
        stop_cmd = "./stop-server.sh".into();
    } else if let Some(f) = files.iter().find(|f| {
        (f.starts_with("start-") && f.ends_with(".sh")) || (f.starts_with("launch_") && f.ends_with(".sh"))
    }) {
        start_cmd = format!("./{f}");
        stop_cmd = String::new();
    }

    // Scan .env* and *.sh for hints, ports and container names.
    let mut hint_text = files.join("\n");
    let mut containers: Vec<String> = Vec::new();
    let mut port: Option<u16> = None;
    for f in &files {
        let is_sh = f.ends_with(".sh");
        let is_env = f.starts_with(".env");
        if !(is_sh || is_env) {
            continue;
        }
        let Some(text) = fs_read(&format!("{path}/{f}")).await else {
            continue;
        };
        hint_text.push('\n');
        hint_text.push_str(&text);
        for line in text.lines() {
            let lt = line.trim();
            if is_env {
                if let Some(rest) = lt.strip_prefix("PORT=") {
                    if let Ok(v) = rest.trim().parse::<u16>() {
                        port = Some(v);
                    }
                }
            }
            let tokens: Vec<&str> = lt.split_whitespace().collect();
            for i in 0..tokens.len() {
                if lt.contains("docker") && tokens[i] == "--name" && i + 1 < tokens.len() {
                    containers.push(tokens[i + 1].to_string());
                }
                if i >= 2 && tokens[i - 2] == "docker" && tokens[i - 1] == "rm" && tokens[i] == "-f" && i + 1 < tokens.len()
                {
                    containers.push(tokens[i + 1].to_string());
                }
            }
        }
    }
    containers.sort();
    containers.dedup();

    let engine = if files.iter().any(|f| f == "launch.sh") && hint_text.contains("tensorfold") {
        "tensorfold"
    } else {
        "vllm"
    };

    let port = port.unwrap_or(8888);
    let lname = name.to_lowercase();
    let slot: Option<&str> = if port == 8888 {
        Some("primary-llm")
    } else if lname.contains("embed") {
        Some("aux-embedding")
    } else if lname.contains("rerank") {
        Some("aux-reranker")
    } else {
        None
    };

    Ok(Json(json!({
        "ok": true,
        "suggestion": {
            "id": slugify(&name),
            "name": name,
            "path": path,
            "start_cmd": start_cmd,
            "stop_cmd": stop_cmd,
            "host": default_host(),
            "port": port,
            "slot": slot,
            "engine": engine,
            "containers": containers,
            "process_match": Value::Null,
        }
    })))
}

async fn delete_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let reg;
    {
        let mut st = lock_state(&state.state);
        let before = st.registry.len();
        st.registry.retain(|e| e.id != id);
        if st.registry.len() == before {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))));
        }
        st.runtime.remove(&id);
        reg = st.registry.clone();
    }
    persist_registry(&reg).await;
    Ok(Json(json!({ "ok": true })))
}

async fn start_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let entry = {
        let st = lock_state(&state.state);
        match st.registry.iter().find(|e| e.id == id) {
            Some(e) => e.clone(),
            None => return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" })))),
        }
    };

    // Gate on shared storage health.
    let (online, reason) = {
        let st = lock_state(&state.state);
        (st.snapshot.storage.online, st.snapshot.storage.reason.clone())
    };
    if !online {
        let msg = format!("Shared storage offline: {}", reason.unwrap_or_else(|| "unknown".into()));
        return Err((StatusCode::CONFLICT, Json(json!({ "error": msg }))));
    }

    // Enforce a single active primary-llm.
    {
        let st = lock_state(&state.state);
        if entry.slot == "primary-llm" {
            let active = st
                .registry
                .iter()
                .filter(|o| o.id != entry.id && o.slot == "primary-llm")
                .find(|o| {
                    matches!(
                        st.runtime.get(&o.id).map(|r| r.status),
                        Some(Status::Running) | Some(Status::Starting)
                    )
                });
            if let Some(o) = active {
                let msg = format!("Stop the active primary LLM first: {}", o.name);
                return Err((StatusCode::CONFLICT, Json(json!({ "error": msg }))));
            }
        }
    }

    {
        let mut st = lock_state(&state.state);
        let rt = st.runtime.entry(entry.id.clone()).or_default();
        rt.status = Status::Starting;
        if rt.running_since_unix.is_none() {
            rt.running_since_unix = Some(now_unix());
        }
    }

    if let Err(e) = spawn_logged_cmd(&entry, &entry.start_cmd).await {
        let mut st = lock_state(&state.state);
        if let Some(rt) = st.runtime.get_mut(&entry.id) {
            rt.status = Status::Stopped;
            rt.running_since_unix = None;
        }
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))));
    }
    Ok(Json(json!({ "ok": true })))
}

async fn stop_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let entry = {
        let st = lock_state(&state.state);
        match st.registry.iter().find(|e| e.id == id) {
            Some(e) => e.clone(),
            None => return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" })))),
        }
    };
    {
        let mut st = lock_state(&state.state);
        let rt = st.runtime.entry(entry.id.clone()).or_default();
        rt.status = Status::Stopping;
    }
    if let Err(e) = spawn_logged_cmd(&entry, &entry.stop_cmd).await {
        eprintln!("[stop] {}: {e}", entry.id);
    }
    Ok(Json(json!({ "ok": true })))
}

async fn entry_logs(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (entry, running): (Entry, Vec<(String, String)>) = {
        let st = lock_state(&state.state);
        let Some(entry) = st.registry.iter().find(|e| e.id == id).cloned() else {
            return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))));
        };
        let running = entry
            .containers
            .iter()
            .filter_map(|c| st.container_node.get(c.as_str()).cloned().map(|node| (c.clone(), node)))
            .collect();
        (entry, running)
    };

    if !running.is_empty() {
        let mut lines: Vec<String> = Vec::new();
        for (container, node) in &running {
            let res = if node == REMOTE_NODE {
                let script = format!("docker logs --tail 100 {container}");
                ssh_ai2(&script, LOGS_TIMEOUT_MS).await
            } else {
                run_checked("docker", &["logs", "--tail", "100", container.as_str()], LOGS_TIMEOUT_MS).await
            };
            lines.push(format!("=== {container} ({node}) ==="));
            match res {
                Ok(out) => {
                    lines.extend(String::from_utf8_lossy(&out.stdout).lines().map(|s| s.to_string()));
                    lines.extend(String::from_utf8_lossy(&out.stderr).lines().map(|s| s.to_string()));
                }
                Err(e) => lines.push(format!("error: {e}")),
            }
        }
        return Ok(Json(json!({ "source": "docker", "lines": lines })));
    }

    // No running containers — fall back to the entry's own log file.
    let log_path = format!("logs/{}.log", entry.id);
    match tokio::fs::read_to_string(&log_path).await {
        Ok(text) => {
            let all: Vec<&str> = text.lines().collect();
            let start = all.len().saturating_sub(200);
            let lines: Vec<String> = all[start..].iter().map(|s| s.to_string()).collect();
            Ok(Json(json!({ "source": "file", "lines": lines })))
        }
        Err(_) => Ok(Json(json!({ "source": "file", "lines": ["(no log file yet)"] }))),
    }
}

async fn drop_caches() -> Json<Value> {
    let local = run_shell("sudo -n sync && sudo -n sysctl -w vm.drop_caches=3", DROP_CACHES_TIMEOUT_MS).await;
    let remote = ssh_ai2("sudo -n sync && sudo -n sysctl -w vm.drop_caches=3", DROP_CACHES_TIMEOUT_MS).await;

    let to_str = |r: &Result<std::process::Output, String>| match r {
        Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
        Err(e) => format!("error: {e}"),
    };
    let ai = to_str(&local);
    let ai2 = to_str(&remote);
    let ok = local.as_ref().map(|o| o.status.success()).unwrap_or(false)
        && remote.as_ref().map(|o| o.status.success()).unwrap_or(false);
    Json(json!({ "ok": ok, "ai": ai, "ai2": ai2 }))
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let _ = tokio::fs::create_dir_all("data").await;
    let _ = tokio::fs::create_dir_all("logs").await;

    let (registry, seeded) = load_registry().await;
    if seeded {
        eprintln!("[boot] seeded registry with {} entries", registry.len());
        persist_registry(&registry).await;
    }

    let mut runtime: HashMap<String, EntryRuntime> = HashMap::new();
    for e in &registry {
        runtime.insert(e.id.clone(), EntryRuntime::default());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .connect_timeout(Duration::from_secs(2))
        .build()
        .expect("failed to build reqwest client");

    let state = Arc::new(AppState {
        state: Mutex::new(DashboardState {
            registry,
            runtime,
            snapshot: Snapshot::default(),
            container_node: HashMap::new(),
        }),
        client,
    });

    tokio::spawn(storage_loop(state.clone()));
    tokio::spawn(metrics_loop(state.clone()));
    tokio::spawn(telemetry_loop(state.clone()));

    let app = Router::new()
        .route("/", get(root))
        .route("/api/state", get(api_state))
        .route("/api/entries", post(add_entry))
        .route("/api/entries/inspect", post(inspect_entry))
        .route("/api/entries/{id}", delete(delete_entry))
        .route("/api/entries/{id}/start", post(start_entry))
        .route("/api/entries/{id}/stop", post(stop_entry))
        .route("/api/entries/{id}/logs", get(entry_logs))
        .route("/api/cluster/drop-caches", post(drop_caches))
        .with_state(state);

    match tokio::net::TcpListener::bind(BIND_ADDR).await {
        Ok(listener) => {
            eprintln!("gx10-dashboard listening on {BIND_ADDR}");
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("server error: {e}");
            }
        }
        Err(e) => {
            eprintln!("failed to bind {BIND_ADDR}: {e}");
            std::process::exit(1);
        }
    }
}
