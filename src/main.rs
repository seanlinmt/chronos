// =============================================================================
// spark-deck — TUI cluster-control / telemetry dashboard for a 2-node
// NVIDIA DGX Spark (GB10) cluster (head: `ai`, worker: `ai2`).
//
// Single-file ratatui app. Background collectors poll HTTP endpoints (2s) and
// system/SSH probes (3s), publishing snapshots through a tokio::sync::watch
// channel to the render loop. `--dump-state` runs one full pass and prints the
// JSON state (verification path).
//
// Sections:
//   1. constants / paths
//   2. data model (registry + JSON snapshots)
//   3. registry load/seed/save
//   4. small parsing helpers
//   5. collectors (storage, nodes, GPU attribution, HTTP probes)
//   6. shared state + publishers
//   7. lifecycle actions (start/stop/add/remove/drop-caches/logs)
//   8. TUI (ui state, drawing, key handling)
//   9. main
// =============================================================================

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};
use tokio::sync::{mpsc, watch};

// =============================================================================
// 1. CONSTANTS / PATHS
// =============================================================================

const AI2_LAN: &str = "192.168.20.229";
const AI_LAN: &str = "192.168.20.63";
const HTTP_TIMEOUT_MS: u64 = 1500;
const HTTP_INTERVAL_SECS: u64 = 2;
const SYS_INTERVAL_SECS: u64 = 3;
const CHECK_TIMEOUT_SECS: u64 = 2; // per storage/system check
const SSH_BLOB_TIMEOUT_SECS: u64 = 6; // one combined ssh batch for ai2
const STARTING_GRACE_SECS: u64 = 300; // STARTING until endpoint answers (or this expires)

/// Base dir for data + logs. On the head node this is /home/sean/spark-deck;
/// fall back to cwd (e.g. when type-checking/running locally on a Mac).
fn base_dir() -> PathBuf {
    let p = PathBuf::from("/home/sean/spark-deck");
    if p.is_dir() {
        p
    } else {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    }
}

fn logs_dir() -> PathBuf {
    base_dir().join("logs")
}

fn registry_path() -> PathBuf {
    base_dir().join("data").join("registry.json")
}

fn log_file_for(id: &str) -> PathBuf {
    logs_dir().join(format!("{}.log", sanitize_id(id)))
}

fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect()
}

fn is_ai2_host(host: &str) -> bool {
    host == AI2_LAN || host == "ai2"
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_millis(HTTP_TIMEOUT_MS))
        .build()
        .expect("failed to build reqwest client")
}

// =============================================================================
// 2. DATA MODEL
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryEntry {
    id: String,
    path: String,
    #[serde(default)]
    start: String,
    #[serde(default)]
    stop: String,
    host: String,
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

fn default_engine() -> String {
    "vllm".to_string()
}

#[derive(Debug, Clone, Serialize, Default)]
struct StorageChecks {
    ai_findmnt: bool,
    ai_mdstat: bool,
    ai_stat: bool,
    ai2_findmnt: bool,
    ai2_stat: bool,
}

#[derive(Debug, Clone, Serialize, Default)]
struct StorageState {
    online: bool,
    fstype: Option<String>,
    reason: Option<String>,
    total_bytes: Option<u64>,
    used_bytes: Option<u64>,
    avail_bytes: Option<u64>,
    checks: StorageChecks,
}

#[derive(Debug, Clone, Serialize, Default)]
struct DiskInfo {
    total_bytes: u64,
    used_bytes: u64,
    avail_bytes: u64,
    use_pct: f64,
}

#[derive(Debug, Clone, Serialize, Default)]
struct NodeState {
    mem_total_bytes: Option<u64>,
    mem_available_bytes: Option<u64>,
    mem_used_bytes: Option<u64>,
    mem_cache_bytes: Option<u64>,
    swap_total_bytes: Option<u64>,
    swap_used_bytes: Option<u64>,
    disk_root: Option<DiskInfo>,
}

/// One entry in the published JSON state (shape fixed by spec).
#[derive(Debug, Clone, Serialize)]
struct EntrySnapshot {
    id: String,
    path: String,
    slot: String,
    engine: String,
    host: String,
    port: u16,
    state: String,
    containers: Vec<String>,
    served_model: Option<String>,
    status_detail: Option<String>,
    mem_ai_bytes: u64,
    mem_ai2_bytes: u64,
    mem_total_bytes: u64,
    concurrent_requests: Option<u64>,
    queued_requests: Option<u64>,
    live_tok_per_sec: Option<f64>,
    lifetime_decode_tok_per_sec: Option<f64>,
    total_tokens_generated: Option<u64>,
    prompt_tokens_total: Option<u64>,
    cached_tokens_total: Option<u64>,
}

/// Top-level JSON document (also the TUI snapshot).
#[derive(Debug, Clone, Serialize, Default)]
struct AppState {
    timestamp: String,
    storage: StorageState,
    nodes: BTreeMap<String, Option<NodeState>>,
    entries: Vec<EntrySnapshot>,
}

/// Rolling counters kept between polls to compute deltas.
#[derive(Debug, Clone, Default)]
struct Prev {
    total: Option<u64>,
    decode_s: Option<f64>,
    at: Option<Instant>,
}

#[derive(Debug, Clone, Default)]
struct ProbeResult {
    ok: bool,
    detail: Option<String>,
    served_model: Option<String>,
    concurrent: Option<u64>,
    queued: Option<u64>,
    live_tok_per_sec: Option<f64>,
    lifetime_decode: Option<f64>,
    total_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum GraceKind {
    Starting,
    Stopping,
}

/// Mutable state shared between collectors, actions and the TUI.
struct Shared {
    tx: watch::Sender<Arc<AppState>>,
    registry: RwLock<Vec<RegistryEntry>>,
    storage: RwLock<StorageState>,
    nodes: RwLock<BTreeMap<String, Option<NodeState>>>,
    /// id -> (mem_ai_bytes, mem_ai2_bytes)
    mem: RwLock<HashMap<String, (u64, u64)>>,
    probes: RwLock<HashMap<String, ProbeResult>>,
    states: RwLock<HashMap<String, String>>,
    grace: Mutex<HashMap<String, (GraceKind, Instant)>>,
}

impl Shared {
    fn publish(&self) {
        let registry = self.registry.read().unwrap().clone();
        let storage = self.storage.read().unwrap().clone();
        let nodes = self.nodes.read().unwrap().clone();
        let mem = self.mem.read().unwrap().clone();
        let probes = self.probes.read().unwrap().clone();
        let states = self.states.read().unwrap().clone();
        let app = assemble(&registry, &storage, &nodes, &mem, &probes, &states);
        let _ = self.tx.send(Arc::new(app));
    }
}

fn assemble(
    registry: &[RegistryEntry],
    storage: &StorageState,
    nodes: &BTreeMap<String, Option<NodeState>>,
    mem: &HashMap<String, (u64, u64)>,
    probes: &HashMap<String, ProbeResult>,
    states: &HashMap<String, String>,
) -> AppState {
    let mut entries = Vec::new();
    for e in registry {
        let (m_ai, m_ai2) = mem.get(&e.id).copied().unwrap_or((0, 0));
        let p = probes.get(&e.id);
        let state = states
            .get(&e.id)
            .cloned()
            .unwrap_or_else(|| "STOPPED".to_string());
        entries.push(EntrySnapshot {
            id: e.id.clone(),
            path: e.path.clone(),
            slot: e.slot.clone(),
            engine: e.engine.clone(),
            host: e.host.clone(),
            port: e.port,
            state,
            containers: e.containers.clone(),
            served_model: p.and_then(|x| x.served_model.clone()),
            status_detail: p.and_then(|x| x.detail.clone()),
            mem_ai_bytes: m_ai,
            mem_ai2_bytes: m_ai2,
            mem_total_bytes: m_ai + m_ai2,
            concurrent_requests: p.and_then(|x| x.concurrent),
            queued_requests: p.and_then(|x| x.queued),
            live_tok_per_sec: p.and_then(|x| x.live_tok_per_sec),
            lifetime_decode_tok_per_sec: p.and_then(|x| x.lifetime_decode),
            total_tokens_generated: p.and_then(|x| x.total_tokens),
            prompt_tokens_total: p.and_then(|x| x.prompt_tokens),
            cached_tokens_total: p.and_then(|x| x.cached_tokens),
        });
    }
    AppState {
        timestamp: chrono::Utc::now().to_rfc3339(),
        storage: storage.clone(),
        nodes: nodes.clone(),
        entries,
    }
}

// =============================================================================
// 3. REGISTRY LOAD / SEED / SAVE
// =============================================================================

fn seed_registry() -> Vec<RegistryEntry> {
    vec![
        RegistryEntry {
            id: "glm53-tensorfold-spark".into(),
            path: "/home/sean/repos/glm53-tensorfold-spark".into(),
            start: "./launch.sh start".into(),
            stop: "./launch.sh stop".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "tensorfold".into(),
            containers: vec!["glm53-tf-r0".into(), "glm53-tf-r1".into()],
            process_match: None,
        },
        RegistryEntry {
            id: "DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            path: "/home/sean/repos/DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            start: "./start-server.sh".into(),
            stop: "./stop-server.sh".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "vllm".into(),
            containers: vec![],
            process_match: None,
        },
        RegistryEntry {
            id: "veloGB10 (Qwen3.8-Flash-Next)".into(),
            path: "/home/sean/repos/veloGB10".into(),
            start: "./launch_qwen38_flash_next.sh 2".into(),
            stop: "pkill -f gb10_inference || true; ssh -o BatchMode=yes ai2 'pkill -f gb10_inference || true'".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "velogb10".into(),
            containers: vec![],
            process_match: Some("gb10_inference".into()),
        },
        RegistryEntry {
            id: "launch-embedding-ai.sh (BGE-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start: "./launch-embedding-ai.sh".into(),
            stop: "docker rm -f vllm_bge_m3".into(),
            host: AI_LAN.into(),
            port: 8891,
            slot: "aux-embedding".into(),
            engine: "vllm".into(),
            containers: vec!["vllm_bge_m3".into()],
            process_match: None,
        },
        RegistryEntry {
            id: "launch-reranker-ai2.sh (BGE-Reranker-v2-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start: "./launch-reranker-ai2.sh".into(),
            stop: "ssh -o BatchMode=yes ai2 'docker rm -f vllm_bge_reranker'".into(),
            host: AI2_LAN.into(),
            port: 8892,
            slot: "aux-reranker".into(),
            engine: "vllm".into(),
            containers: vec!["vllm_bge_reranker".into()],
            process_match: None,
        },
    ]
}

fn load_registry() -> Vec<RegistryEntry> {
    let p = registry_path();
    if let Ok(s) = std::fs::read_to_string(&p) {
        if let Ok(v) = serde_json::from_str::<Vec<RegistryEntry>>(&s) {
            return v;
        }
    }
    let seeded = seed_registry();
    let _ = save_registry(&seeded);
    seeded
}

fn save_registry(entries: &[RegistryEntry]) -> Result<(), String> {
    let p = registry_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let s = serde_json::to_string_pretty(entries).map_err(|e| e.to_string())?;
    std::fs::write(&p, s).map_err(|e| format!("write {}: {e}", p.display()))
}

// =============================================================================
// 4. SMALL PARSING / FORMATTING HELPERS
// =============================================================================

/// df -B1 <path> -> (total, used, avail, use_pct)
fn parse_df(output: &str) -> Option<(u64, u64, u64, f64)> {
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() < 2 {
        return None;
    }
    let f: Vec<&str> = lines[1].split_whitespace().collect();
    if f.len() < 5 {
        return None;
    }
    let total = f[1].parse().ok()?;
    let used = f[2].parse().ok()?;
    let avail = f[3].parse().ok()?;
    let pct = f[4].trim_end_matches('%').parse::<f64>().ok()?;
    Some((total, used, avail, pct))
}

fn meminfo_key(s: &str, key: &str) -> Option<u64> {
    s.lines()
        .find(|l| l.starts_with(key))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// /proc/meminfo -> (total_kb, avail_kb, cache_kb, swap_total_kb, swap_used_kb)
fn parse_meminfo(s: &str) -> Option<(u64, u64, u64, u64, u64)> {
    let total = meminfo_key(s, "MemTotal:")?;
    let avail = meminfo_key(s, "MemAvailable:").unwrap_or_else(|| meminfo_key(s, "MemFree:").unwrap_or(0));
    let buffers = meminfo_key(s, "Buffers:").unwrap_or(0);
    let cached = meminfo_key(s, "Cached:").unwrap_or(0);
    let st = meminfo_key(s, "SwapTotal:").unwrap_or(0);
    let sf = meminfo_key(s, "SwapFree:").unwrap_or(0);
    Some((total, avail, buffers + cached, st, st.saturating_sub(sf)))
}

/// nvidia-smi csv "pid, mib" -> (pid, bytes)
fn parse_gpu_apps(s: &str) -> Vec<(u32, u64)> {
    s.lines()
        .filter_map(|l| {
            let mut it = l.split(',');
            let pid: u32 = it.next()?.trim().parse().ok()?;
            let mib: u64 = it.next()?.trim().parse().ok()?;
            Some((pid, mib.saturating_mul(1_048_576)))
        })
        .collect()
}

/// docker ps --format '{{.ID}}|{{.Names}}' -> {id -> name}
fn parse_docker_ps(s: &str) -> HashMap<String, String> {
    s.lines()
        .filter_map(|l| {
            let (id, name) = l.split_once('|')?;
            if id.trim().len() >= 12 {
                Some((id.trim().to_string(), name.trim().to_string()))
            } else {
                None
            }
        })
        .collect()
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
}

fn compact(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn jnum(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_f64()).map(|f| f as u64)
}

fn jf(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

/// Split a marker-delimited blob ("===NAME===") into sections.
fn sections(s: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut name = String::new();
    let mut cur = String::new();
    for line in s.lines() {
        if line.starts_with("===") && line.ends_with("===") {
            if !name.is_empty() {
                map.insert(name.clone(), cur.clone());
            }
            name = line.trim_matches('=').to_string();
            cur.clear();
        } else {
            cur.push_str(line);
            cur.push('\n');
        }
    }
    if !name.is_empty() {
        map.insert(name, cur);
    }
    map
}

#[derive(Debug, Clone, Default)]
struct ProcInfo {
    pid: u32,
    rss_bytes: u64,
    container_hex: Option<String>,
    cmdline: String,
}

// =============================================================================
// 5. COLLECTORS
// =============================================================================

async fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    match tokio::process::Command::new(program).args(args).output().await {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).to_string()),
        _ => None,
    }
}

async fn ssh_ai2(cmd: &str) -> Result<String, String> {
    let out = tokio::process::Command::new("ssh")
        .args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "ai2", cmd])
        .output()
        .await
        .map_err(|e| format!("ssh spawn failed: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "ssh exit {:?}: {}",
            out.status.code(),
            err.lines().next().unwrap_or("unknown error")
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// One combined SSH batch for ai2 (meminfo, df, storage checks, GPU apps,
/// docker ps, per-PID proc samples). Minimizes SSH calls per tick.
const AI2_BLOB: &str = "echo '===MEMINFO==='; grep -E '^(MemTotal|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):' /proc/meminfo; \
echo '===DFROOT==='; df -B1 /; \
echo '===FINDMNT==='; findmnt -n -o FSTYPE,OPTIONS /mnt/nvmeof; \
echo '===DFNVMEOF==='; df -B1 /mnt/nvmeof; \
echo '===STAT==='; stat -t /mnt/nvmeof/repos /mnt/nvmeof/huggingface >/dev/null 2>&1 && echo OK || echo FAIL; \
echo '===GPUAPPS==='; nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null; \
echo '===DOCKERPS==='; docker ps --no-trunc --format '{{.ID}}|{{.Names}}' 2>/dev/null; \
echo '===PROCS==='; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do \
rss=$(grep VmRSS /proc/$p/status 2>/dev/null | awk '{print $2}'); cg=$(tail -1 /proc/$p/cgroup 2>/dev/null); \
cm=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200); echo \"PROC $p ${rss:-0} $cg\"; echo \"CMD $cm\"; done; \
echo '===END==='";

/// Storage gatekeeper pass (ai local + ai2 via ssh), every check timeout-wrapped.
async fn collect_storage() -> StorageState {
    let mut checks = StorageChecks::default();
    let mut reasons: Vec<String> = Vec::new();

    // --- ai (local) ---
    let fm = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        run_cmd("findmnt", &["-n", "-o", "FSTYPE,OPTIONS", "/mnt/nvmeof"]),
    )
    .await;
    checks.ai_findmnt = matches!(fm, Ok(Some(s)) if s.contains("gfs2"));
    if !checks.ai_findmnt {
        reasons.push("ai: findmnt /mnt/nvmeof not gfs2 (or timed out)".into());
    }

    let md = tokio::time::timeout(Duration::from_secs(CHECK_TIMEOUT_SECS), async {
        tokio::fs::read_to_string("/proc/mdstat").await.map_err(|e| e.to_string())
    })
    .await;
    checks.ai_mdstat = matches!(&md, Ok(Ok(s)) if s.contains("md0 : active raid0"));
    if !checks.ai_mdstat {
        reasons.push("ai: /proc/mdstat missing 'md0 : active raid0'".into());
    }

    let st = tokio::time::timeout(Duration::from_secs(CHECK_TIMEOUT_SECS), async {
        tokio::process::Command::new("stat")
            .args(["/mnt/nvmeof/repos", "/mnt/nvmeof/huggingface"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .map_err(|e| e.to_string())
    })
    .await;
    checks.ai_stat = matches!(st, Ok(Ok(true)));
    if !checks.ai_stat {
        reasons.push("ai: stat /mnt/nvmeof/{repos,huggingface} failed (GFS2 hang?)".into());
    }

    // df for the shared GFS2 fs (only meaningful when gate passes; use ai's view).
    let df_nvmeof = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        run_cmd("df", &["-B1", "/mnt/nvmeof"]),
    )
    .await
    .ok()
    .flatten()
    .and_then(|s| parse_df(&s));

    // --- ai2 (via one combined ssh blob) ---
    let blob = tokio::time::timeout(
        Duration::from_secs(SSH_BLOB_TIMEOUT_SECS),
        ssh_ai2(AI2_BLOB),
    )
    .await;
    let sec = match blob {
        Ok(Ok(s)) => sections(&s),
        Ok(Err(e)) => {
            reasons.push(format!("ai2: unreachable ({e})"));
            HashMap::new()
        }
        Err(_) => {
            reasons.push("ai2: ssh blob timed out".into());
            HashMap::new()
        }
    };
    checks.ai2_findmnt = sec.get("FINDMNT").map(|s| s.contains("gfs2")).unwrap_or(false);
    if !checks.ai2_findmnt {
        reasons.push("ai2: findmnt /mnt/nvmeof not gfs2".into());
    }
    checks.ai2_stat = sec.get("STAT").map(|s| s.contains("OK")).unwrap_or(false);
    if !checks.ai2_stat {
        reasons.push("ai2: stat /mnt/nvmeof/{repos,huggingface} failed".into());
    }

    let online = checks.ai_findmnt && checks.ai_mdstat && checks.ai_stat && checks.ai2_findmnt && checks.ai2_stat;
    let df = df_nvmeof;

    StorageState {
        online,
        fstype: if online { Some("gfs2".into()) } else { None },
        reason: if online { None } else { Some(reasons.join("; ")) },
        total_bytes: df.map(|d| d.0),
        used_bytes: df.map(|d| d.1),
        avail_bytes: df.map(|d| d.2),
        checks,
    }
}

async fn node_from_blob(blob: &str) -> NodeState {
    let sec = sections(blob);
    let mem = sec.get("MEMINFO").and_then(|s| parse_meminfo(s));
    let disk = sec.get("DFROOT").and_then(|s| parse_df(s));
    NodeState {
        mem_total_bytes: mem.map(|m| m.0 * 1024),
        mem_available_bytes: mem.map(|m| m.1 * 1024),
        mem_used_bytes: mem.map(|m| m.0.saturating_sub(m.1) * 1024),
        mem_cache_bytes: mem.map(|m| m.2 * 1024),
        swap_total_bytes: mem.map(|m| m.3 * 1024),
        swap_used_bytes: mem.map(|m| m.4 * 1024),
        disk_root: disk.map(|d| DiskInfo {
            total_bytes: d.0,
            used_bytes: d.1,
            avail_bytes: d.2,
            use_pct: d.3,
        }),
    }
}

async fn collect_nodes() -> BTreeMap<String, Option<NodeState>> {
    let mut out = BTreeMap::new();

    // ai — local reads, each timeout-wrapped.
    let meminfo = tokio::time::timeout(Duration::from_secs(CHECK_TIMEOUT_SECS), async {
        tokio::fs::read_to_string("/proc/meminfo").await.map_err(|e| e.to_string())
    })
    .await;
    let df_root = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        run_cmd("df", &["-B1", "/"]),
    )
    .await;
    let ai_node = match (meminfo, df_root) {
        (Ok(Ok(m)), Ok(df)) => {
            let mem = parse_meminfo(&m);
            let disk = df.as_deref().and_then(parse_df);
            Some(NodeState {
                mem_total_bytes: mem.map(|m| m.0 * 1024),
                mem_available_bytes: mem.map(|m| m.1 * 1024),
                mem_used_bytes: mem.map(|m| m.0.saturating_sub(m.1) * 1024),
                mem_cache_bytes: mem.map(|m| m.2 * 1024),
                swap_total_bytes: mem.map(|m| m.3 * 1024),
                swap_used_bytes: mem.map(|m| m.4 * 1024),
                disk_root: disk.map(|d| DiskInfo {
                    total_bytes: d.0,
                    used_bytes: d.1,
                    avail_bytes: d.2,
                    use_pct: d.3,
                }),
            })
        }
        _ => None,
    };
    out.insert("ai".into(), ai_node);

    // ai2 — single combined ssh blob.
    let blob = tokio::time::timeout(Duration::from_secs(SSH_BLOB_TIMEOUT_SECS), ssh_ai2(AI2_BLOB)).await;
    out.insert(
        "ai2".into(),
        match blob {
            Ok(Ok(s)) => Some(node_from_blob(&s).await),
            _ => None,
        },
    );

    out
}

fn proc_rss(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let kb: u64 = s
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

fn proc_container(pid: u32) -> Option<String> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?;
    let line = s.lines().find(|l| l.contains("docker-"))?;
    let rest = line.split("docker-").nth(1)?;
    let id = rest.trim_end_matches(".scope").trim();
    if id.len() == 64 {
        Some(id.to_string())
    } else {
        None
    }
}

fn proc_cmdline(pid: u32) -> String {
    std::fs::read(format!("/proc/{pid}/cmdline"))
        .map(|b| String::from_utf8_lossy(&b).replace('\0', " "))
        .unwrap_or_default()
}

/// Fallback substring used when an entry has neither containers nor
/// process_match (e.g. "DeepSeek-v4.1-Flash-2x-DGX-Spark" -> "deepseek").
fn entry_fallback_match(e: &RegistryEntry) -> String {
    let first = e
        .id
        .split(|c: char| c == '-' || c == ' ' || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if first.len() >= 4 {
        first
    } else {
        e.id.to_ascii_lowercase()
    }
}

fn entry_matches_pid(e: &RegistryEntry, name: Option<&str>, cmdline: &str) -> bool {
    let cl = cmdline.to_ascii_lowercase();
    if let Some(hex) = name {
        if e.containers.iter().any(|c| c == hex) {
            return true;
        }
    }
    if let Some(pm) = &e.process_match {
        if cl.contains(&pm.to_ascii_lowercase()) {
            return true;
        }
    }
    if e.containers.is_empty() && e.process_match.is_none() {
        return cl.contains(&entry_fallback_match(e));
    }
    false
}

fn attribute_pids(
    gpu: Vec<(u32, u64)>,
    docker: &HashMap<String, String>,
    procs: Vec<ProcInfo>,
    registry: &[RegistryEntry],
) -> HashMap<String, u64> {
    let by_pid: HashMap<u32, &ProcInfo> = procs.iter().map(|p| (p.pid, p)).collect();
    let mut out: HashMap<String, u64> = HashMap::new();
    for (pid, gpu_bytes) in gpu {
        let rss = by_pid.get(&pid).map(|p| p.rss_bytes).unwrap_or(0);
        let total = gpu_bytes + rss;
        let cmdline = by_pid.get(&pid).map(|p| p.cmdline.as_str()).unwrap_or("");
        let cname = by_pid
            .get(&pid)
            .and_then(|p| p.container_hex.as_ref())
            .and_then(|hex| docker.get(hex))
            .map(|s| s.as_str());
        for e in registry {
            if entry_matches_pid(e, cname, cmdline) {
                *out.entry(e.id.clone()).or_insert(0) += total;
                break; // attribute each pid to exactly one entry
            }
        }
    }
    out
}

/// Parse the ===PROCS=== / ===CMD=== section emitted by AI2_BLOB.
fn parse_remote_procs(s: &str) -> Vec<ProcInfo> {
    let mut out: Vec<ProcInfo> = Vec::new();
    let mut cur: Option<ProcInfo> = None;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("PROC ") {
            if let Some(p) = cur.take() {
                out.push(p);
            }
            let mut it = rest.split_whitespace();
            let pid: u32 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let rss: u64 = it.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let cg = it.collect::<Vec<_>>().join(" ");
            let container_hex = if cg.contains("docker-") {
                cg.split("docker-")
                    .nth(1)
                    .map(|x| x.trim_end_matches(".scope").trim())
                    .filter(|id| id.len() == 64)
                    .map(|id| id.to_string())
            } else {
                None
            };
            cur = Some(ProcInfo {
                pid,
                rss_bytes: rss * 1024,
                container_hex,
                cmdline: String::new(),
            });
        } else if let Some(rest) = line.strip_prefix("CMD ") {
            if let Some(p) = cur.as_mut() {
                p.cmdline = rest.to_string();
            }
        }
    }
    if let Some(p) = cur.take() {
        out.push(p);
    }
    out
}

async fn collect_mem_attr(registry: &[RegistryEntry]) -> HashMap<String, (u64, u64)> {
    // --- ai (local) ---
    let gpu_ai = run_cmd(
        "nvidia-smi",
        &["--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
    )
    .await
    .unwrap_or_default();
    let docker_ai = run_cmd("docker", &["ps", "--no-trunc", "--format", "{{.ID}}|{{.Names}}"])
        .await
        .unwrap_or_default();
    let procs_ai: Vec<ProcInfo> = parse_gpu_apps(&gpu_ai)
        .iter()
        .map(|(pid, _)| ProcInfo {
            pid: *pid,
            rss_bytes: proc_rss(*pid).unwrap_or(0),
            container_hex: proc_container(*pid),
            cmdline: proc_cmdline(*pid),
        })
        .collect();
    let docker_map = parse_docker_ps(&docker_ai);
    let ai_map = attribute_pids(parse_gpu_apps(&gpu_ai), &docker_map, procs_ai, registry);

    // --- ai2 (from the same ssh blob used elsewhere; re-run it here) ---
    let ai2_map = match tokio::time::timeout(Duration::from_secs(SSH_BLOB_TIMEOUT_SECS), ssh_ai2(AI2_BLOB)).await {
        Ok(Ok(blob)) => {
            let sec = sections(&blob);
            let gpu = sec.get("GPUAPPS").map(|s| parse_gpu_apps(s)).unwrap_or_default();
            let docker = sec
                .get("DOCKERPS")
                .map(|s| parse_docker_ps(s))
                .unwrap_or_default();
            let procs = sec.get("PROCS").map(|s| parse_remote_procs(s)).unwrap_or_default();
            attribute_pids(gpu, &docker, procs, registry)
        }
        _ => HashMap::new(),
    };

    let mut ids: Vec<String> = registry.iter().map(|e| e.id.clone()).collect();
    ids.extend(ai_map.keys().cloned());
    ids.extend(ai2_map.keys().cloned());
    let mut out = HashMap::new();
    for id in ids {
        let a = ai_map.get(&id).copied().unwrap_or(0);
        let b = ai2_map.get(&id).copied().unwrap_or(0);
        out.insert(id, (a, b));
    }
    out
}

// --------------------------- HTTP telemetry ----------------------------------

#[derive(Debug, Default)]
struct CollectorState {
    prev: HashMap<String, Prev>,
}

async fn probe_entry(e: &RegistryEntry, prev: &mut Prev, client: &reqwest::Client) -> ProbeResult {
    let url = format!("http://{}:{}", e.host, e.port);
    match e.engine.as_str() {
        "tensorfold" => probe_tensorfold(e, &url, prev, client).await,
        "velogb10" => probe_velogb10(e, &url, prev, client).await,
        _ => probe_vllm(e, &url, prev, client).await,
    }
}

fn fail_probe(msg: String) -> ProbeResult {
    ProbeResult {
        ok: false,
        detail: Some(msg),
        ..Default::default()
    }
}

async fn probe_tensorfold(
    _e: &RegistryEntry,
    url: &str,
    prev: &mut Prev,
    client: &reqwest::Client,
) -> ProbeResult {
    let now = Instant::now();
    let v: Value = match client.get(format!("{url}/health")).send().await {
        Ok(r) => match r.json().await {
            Ok(v) => v,
            Err(e) => return fail_probe(format!("health json: {e}")),
        },
        Err(e) => return fail_probe(format!("http: {e}")),
    };
    let ok = v.get("ok").and_then(|x| x.as_bool()).unwrap_or(false);
    if !ok {
        return fail_probe("health ok=false".into());
    }
    let total = jnum(&v, "completion_tokens_total");
    let decode = jf(&v, "decode_seconds_total");
    let concurrent = jnum(&v, "requests_running").or_else(|| jnum(&v, "inflight"));
    let live = match (prev.total, total, prev.decode_s, decode) {
        (Some(pt), Some(nt), Some(pd), Some(nd)) if nd > pd => Some((nt - pt) as f64 / (nd - pd)),
        (Some(_), Some(_), _, _) => Some(0.0),
        _ => None,
    };
    let lifetime = match (total, decode) {
        (Some(t), Some(d)) if d > 0.0 => Some(t as f64 / d),
        _ => None,
    };
    let p = std::mem::take(prev);
    *prev = Prev {
        total,
        decode_s: decode,
        at: if p.at.is_some() { p.at } else { Some(now) },
    };
    ProbeResult {
        ok: true,
        detail: None,
        served_model: None,
        concurrent,
        queued: None,
        live_tok_per_sec: live,
        lifetime_decode: lifetime,
        total_tokens: total,
        prompt_tokens: jnum(&v, "prompt_tokens_total"),
        cached_tokens: jnum(&v, "cached_tokens_total"),
    }
}

fn parse_prom_metric(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .find_map(|l| {
            let mut it = l.split_whitespace();
            if it.next() == Some(key) {
                it.next().and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
}

async fn probe_vllm(_e: &RegistryEntry, url: &str, prev: &mut Prev, client: &reqwest::Client) -> ProbeResult {
    let now = Instant::now();
    // Liveness + served model name. NOTE: must check HTTP status — another
    // engine may share this host:port and answer 401 with a JSON error body
    // (parses fine but is NOT liveness).
    let models: Value = match client.get(format!("{url}/v1/models")).send().await {
        Ok(r) => {
            let status = r.status();
            if !status.is_success() {
                return fail_probe(format!("models http {status}"));
            }
            match r.json().await {
                Ok(v) => v,
                Err(e) => return fail_probe(format!("models json: {e}")),
            }
        }
        Err(err) => return fail_probe(format!("http: {err}")),
    };
    let served = models
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    // Metrics are best-effort (may be disabled); only trust a 2xx body.
    let metrics: Option<String> = match client.get(format!("{url}/metrics")).send().await {
        Ok(r) if r.status().is_success() => r.text().await.ok(),
        _ => None,
    };
    let (concurrent, queued, total) = match &metrics {
        Some(txt) => {
            let run = parse_prom_metric(txt, "vllm:num_requests_running").map(|f| f as u64);
            let wait = parse_prom_metric(txt, "vllm:num_requests_waiting").map(|f| f as u64);
            let tot = parse_prom_metric(txt, "vllm:generation_tokens_total").map(|f| f as u64);
            let fallback_conc = if txt.contains("http_requests_total") { Some(0) } else { None };
            (run.or(fallback_conc), wait, tot)
        }
        None => (None, None, None),
    };
    let live = match (prev.total, total) {
        (Some(pt), Some(nt)) if nt > pt => {
            let dt = prev.at.map(|a| now.duration_since(a).as_secs_f64()).unwrap_or(0.0);
            if dt > 0.0 {
                Some((nt - pt) as f64 / dt)
            } else {
                None
            }
        }
        _ => None,
    };
    *prev = Prev {
        total,
        decode_s: prev.decode_s,
        at: Some(now),
    };
    ProbeResult {
        ok: true,
        detail: if metrics.is_some() {
            None
        } else {
            Some("stats disabled; liveness via /v1/models".into())
        },
        served_model: served,
        concurrent,
        queued,
        live_tok_per_sec: live,
        lifetime_decode: None,
        total_tokens: total,
        prompt_tokens: None,
        cached_tokens: None,
    }
}

async fn probe_velogb10(
    _e: &RegistryEntry,
    url: &str,
    prev: &mut Prev,
    client: &reqwest::Client,
) -> ProbeResult {
    let now = Instant::now();
    let v: Value = match client.get(format!("{url}/health")).send().await {
        Ok(r) => match r.json().await {
            Ok(v) => v,
            Err(e) => return fail_probe(format!("health json: {e}")),
        },
        Err(e) => return fail_probe(format!("http: {e}")),
    };
    let ok = v.get("status").and_then(|x| x.as_str()) == Some("ok");
    if !ok {
        return fail_probe("status != ok".into());
    }
    let mtp = &v["telemetry"]["mtp"];
    let df = &v["telemetry"]["dflash2"];
    let mtp_emitted = mtp.get("emitted").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
    let df_emitted = df.get("emitted").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
    let total = Some(mtp_emitted + df_emitted);
    let step = v["telemetry"]["step_ms_p50"].as_f64();
    let yield_sum = mtp.get("yield_tok_per_step").and_then(|x| x.as_f64()).unwrap_or(0.0)
        + df.get("yield_tok_per_step").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let live = match (prev.total, total) {
        (Some(pt), Some(nt)) if nt > pt => {
            let dt = prev.at.map(|a| now.duration_since(a).as_secs_f64()).unwrap_or(0.0);
            if dt > 0.0 {
                Some((nt - pt) as f64 / dt)
            } else {
                None
            }
        }
        _ => {
            if let Some(s) = step {
                if s > 0.0 && yield_sum > 0.0 {
                    Some(yield_sum * 1000.0 / s)
                } else {
                    None
                }
            } else {
                None
            }
        }
    };
    *prev = Prev {
        total,
        decode_s: prev.decode_s,
        at: Some(now),
    };
    ProbeResult {
        ok: true,
        detail: None,
        served_model: None,
        concurrent: None,
        queued: None,
        live_tok_per_sec: live,
        lifetime_decode: None,
        total_tokens: total,
        prompt_tokens: None,
        cached_tokens: None,
    }
}

async fn probe_all(
    registry: &[RegistryEntry],
    prev: &mut HashMap<String, Prev>,
    client: &reqwest::Client,
) -> HashMap<String, ProbeResult> {
    // Run all probes concurrently; each gets its own prev cell.
    let mut handles = Vec::new();
    for e in registry {
        let id = e.id.clone();
        let e = e.clone();
        let mut p = prev.remove(&id).unwrap_or_default();
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            let r = probe_entry(&e, &mut p, &client).await;
            (id, p, r)
        }));
    }
    let mut out = HashMap::new();
    for h in handles {
        if let Ok((id, p, r)) = h.await {
            prev.insert(id.clone(), p);
            out.insert(id, r);
        }
    }
    out
}

// =============================================================================
// 6. SHARED-STATE PUBLISHING LOOPS
// =============================================================================

async fn http_loop(shared: Arc<Shared>, client: reqwest::Client) {
    let mut cs = CollectorState::default();
    let mut interval = tokio::time::interval(Duration::from_secs(HTTP_INTERVAL_SECS));
    loop {
        interval.tick().await;
        let registry = shared.registry.read().unwrap().clone();
        let results = probe_all(&registry, &mut cs.prev, &client).await;

        // Resolve graces whose endpoint state is now known.
        {
            let grace = shared.grace.lock().unwrap();
            let mut resolved = Vec::new();
            for (id, (kind, _)) in grace.iter() {
                let ok = results.get(id).map(|r| r.ok).unwrap_or(false);
                if (matches!(kind, GraceKind::Starting) && ok)
                    || (matches!(kind, GraceKind::Stopping) && !ok)
                {
                    resolved.push(id.clone());
                }
            }
            drop(grace);
            let mut g = shared.grace.lock().unwrap();
            for id in resolved {
                g.remove(&id);
            }
        }

        // Derive per-entry state (RUNNING / STARTING / STOPPED) from probe + grace.
        let states: HashMap<String, String> = {
            let grace = shared.grace.lock().unwrap();
            registry
                .iter()
                .map(|e| {
                    let ok = results.get(&e.id).map(|r| r.ok).unwrap_or(false);
                    let state = if ok {
                        "RUNNING"
                    } else if let Some((GraceKind::Starting, at)) = grace.get(&e.id) {
                        if at.elapsed() < Duration::from_secs(STARTING_GRACE_SECS) {
                            "STARTING"
                        } else {
                            "STOPPED"
                        }
                    } else {
                        "STOPPED"
                    };
                    (e.id.clone(), state.to_string())
                })
                .collect()
        };
        *shared.probes.write().unwrap() = results;
        *shared.states.write().unwrap() = states;
        shared.publish();
    }
}

async fn sys_loop(shared: Arc<Shared>) {
    let mut interval = tokio::time::interval(Duration::from_secs(SYS_INTERVAL_SECS));
    loop {
        interval.tick().await;
        let storage = collect_storage().await;
        let nodes = collect_nodes().await;
        let registry = shared.registry.read().unwrap().clone();
        let mem = collect_mem_attr(&registry).await;
        *shared.storage.write().unwrap() = storage;
        *shared.nodes.write().unwrap() = nodes;
        *shared.mem.write().unwrap() = mem;
        shared.publish();
    }
}

// =============================================================================
// 7. LIFECYCLE ACTIONS
// =============================================================================

enum Action {
    Start(String),
    Stop(String),
    Add(RegistryEntry),
    Remove(String),
    DropCaches,
    ShowLog(String),
}

enum UiMsg {
    Message(String),
    Log { id: String, lines: Vec<String> },
}

use tokio::process::Command as TokioCommand;

/// Spawn a lifecycle command fully detached (setsid, stdin null, output
/// appended to the entry's log file) so quitting the TUI never kills children.
async fn spawn_detached(cwd: &str, cmd: &str, log_path: &Path) -> Result<(), String> {
    use std::process::Stdio;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| format!("open log {}: {e}", log_path.display()))?;
    let stderr = file
        .try_clone()
        .map_err(|e| format!("clone log handle: {e}"))?;
    TokioCommand::new("sh")
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

async fn run_action(action: Action, shared: &Arc<Shared>, ui_tx: &mpsc::UnboundedSender<UiMsg>) {
    match action {
        Action::Start(id) => {
            let entry = shared.registry.read().unwrap().iter().find(|e| e.id == id).cloned();
            let Some(e) = entry else { return };
            let log = log_file_for(&e.id);
            let msg = match spawn_detached(&e.path, &e.start, &log).await {
                Ok(()) => {
                    shared
                        .grace
                        .lock()
                        .unwrap()
                        .insert(e.id.clone(), (GraceKind::Starting, Instant::now()));
                    format!("started `{}` (logs: {})", e.id, log.display())
                }
                Err(err) => format!("start failed: {err}"),
            };
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::Stop(id) => {
            let entry = shared.registry.read().unwrap().iter().find(|e| e.id == id).cloned();
            let Some(e) = entry else { return };
            let log = log_file_for(&e.id);
            let msg = match spawn_detached(&e.path, &e.stop, &log).await {
                Ok(()) => {
                    shared
                        .grace
                        .lock()
                        .unwrap()
                        .insert(e.id.clone(), (GraceKind::Stopping, Instant::now()));
                    format!("stopping `{}` (logs: {})", e.id, log.display())
                }
                Err(err) => format!("stop failed: {err}"),
            };
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::Add(entry) => {
            let res = {
                let mut reg = shared.registry.write().unwrap();
                if reg.iter().any(|e| e.id == entry.id) {
                    Err(format!("entry `{}` already exists", entry.id))
                } else {
                    reg.push(entry.clone());
                    save_registry(&reg).map(|_| format!("added `{}` to registry", entry.id))
                }
            };
            // Next publish tick picks up the new entry.
            let _ = ui_tx.send(UiMsg::Message(match res {
                Ok(m) => m,
                Err(e) => e,
            }));
        }
        Action::Remove(id) => {
            let res = {
                let mut reg = shared.registry.write().unwrap();
                let before = reg.len();
                reg.retain(|e| e.id != id);
                if reg.len() == before {
                    Err(format!("entry `{id}` not found"))
                } else {
                    // never deletes files, only the registry record
                    save_registry(&reg).map(|_| format!("removed `{id}` from registry (files kept)"))
                }
            };
            let _ = ui_tx.send(UiMsg::Message(match res {
                Ok(m) => m,
                Err(e) => e,
            }));
        }
        Action::DropCaches => {
            let local = tokio::process::Command::new("sh")
                .args(["-c", "sudo -n sync && sudo -n sysctl -w vm.drop_caches=3"])
                .output()
                .await;
            let remote = tokio::time::timeout(
                Duration::from_secs(5),
                ssh_ai2("sudo -n sync && sudo -n sysctl -w vm.drop_caches=3"),
            )
            .await;
            let msg = format!(
                "drop_caches: ai {} | ai2 {}",
                match &local {
                    Ok(o) if o.status.success() => "ok".to_string(),
                    Ok(_) => "failed (needs passwordless sudo)".to_string(),
                    Err(e) => format!("error: {e}"),
                },
                match remote {
                    Ok(Ok(_)) => "ok".to_string(),
                    Ok(Err(e)) => format!("failed ({e})"),
                    Err(_) => "timed out".to_string(),
                }
            );
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::ShowLog(id) => {
            let entry = shared.registry.read().unwrap().iter().find(|e| e.id == id).cloned();
            let Some(e) = entry else { return };
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
                let cmd = if is_ai2_host(&e.host) {
                    format!("ssh -o BatchMode=yes -o ConnectTimeout=3 ai2 'docker logs --tail 50 {c}'")
                } else {
                    format!("docker logs --tail 50 {c}")
                };
                match tokio::time::timeout(Duration::from_secs(5), async {
                    tokio::process::Command::new("sh").arg("-c").arg(&cmd).output().await
                })
                .await
                {
                    Ok(Ok(o)) => {
                        lines.push(String::from_utf8_lossy(&o.stdout).to_string());
                        let err = String::from_utf8_lossy(&o.stderr);
                        if !err.trim().is_empty() {
                            lines.push(err.to_string());
                        }
                    }
                    _ => lines.push("(docker logs failed/timed out)".into()),
                }
            }
            let _ = ui_tx.send(UiMsg::Log { id: e.id, lines });
        }
    }
}

async fn action_loop(
    mut rx: mpsc::UnboundedReceiver<Action>,
    shared: Arc<Shared>,
    ui_tx: mpsc::UnboundedSender<UiMsg>,
) {
    while let Some(a) = rx.recv().await {
        run_action(a, &shared, &ui_tx).await;
    }
}

// =============================================================================
// 8. TUI
// =============================================================================

#[derive(Default)]
struct AddModal {
    path: String,
    id: String,
    port: String,
    slot: String,
    engine: String,
    focus: usize,
    note: Option<String>,
    detected_start: Option<String>,
    detected_stop: Option<String>,
}

impl AddModal {
    fn new() -> Self {
        AddModal {
            path: String::new(),
            id: String::new(),
            port: "8888".into(),
            slot: "primary-llm".into(),
            engine: "vllm".into(),
            focus: 0,
            note: None,
            detected_start: None,
            detected_stop: None,
        }
    }
    fn fields(&mut self) -> [&mut String; 5] {
        [
            &mut self.path,
            &mut self.id,
            &mut self.port,
            &mut self.slot,
            &mut self.engine,
        ]
    }
    fn build(&self) -> RegistryEntry {
        RegistryEntry {
            id: if self.id.trim().is_empty() {
                "unnamed-entry".into()
            } else {
                self.id.trim().to_string()
            },
            path: self.path.trim().to_string(),
            start: self.detected_start.clone().unwrap_or_default(),
            stop: self.detected_stop.clone().unwrap_or_default(),
            host: if self.path.starts_with("/mnt/nvmeof") {
                AI2_LAN
            } else {
                AI_LAN
            }
            .to_string(),
            port: self.port.trim().parse().unwrap_or(8888),
            slot: self.slot.trim().to_string(),
            engine: self.engine.trim().to_string(),
            containers: vec![],
            process_match: None,
        }
    }
}

enum Confirm {
    Stop(String),
    Remove(String),
}

#[derive(Default)]
struct UiState {
    selected: usize,
    message: Option<(String, Instant)>,
    should_quit: bool,
    confirm: Option<Confirm>,
    add_modal: Option<AddModal>,
    log_view: Option<(String, Vec<String>)>,
}

// --- add-modal script/env auto-detection ------------------------------------

fn detect_launch(path: &str) -> (Option<String>, Option<String>, Option<String>, Option<String>) {
    let dir = Path::new(path);
    let names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    let mut port = None;
    let mut model = None;
    for envname in names.iter().filter(|n| n.starts_with(".env")) {
        if let Ok(txt) = std::fs::read_to_string(dir.join(envname)) {
            for line in txt.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim().to_ascii_uppercase();
                    let v = v.trim();
                    if (k.contains("PORT") || k == "PORT") && port.is_none() {
                        port = Some(v.to_string());
                    }
                    if (k.contains("MODEL") || k.contains("SERVED")) && model.is_none() {
                        model = Some(v.to_string());
                    }
                }
            }
        }
    }
    let has = |n: &str| names.iter().any(|x| x == n);
    let first_prefix = |p: &str| {
        names
            .iter()
            .filter(|x| x.starts_with(p))
            .min()
            .cloned()
    };
    let (start, stop) = if has("launch.sh") {
        (
            Some("./launch.sh start".to_string()),
            Some("./launch.sh stop".to_string()),
        )
    } else if has("start-server.sh") {
        (
            Some("./start-server.sh".to_string()),
            Some(if has("stop-server.sh") {
                "./stop-server.sh".to_string()
            } else {
                String::new()
            }),
        )
    } else if let Some(s) = first_prefix("start-") {
        (Some(format!("./{s}")), Some(String::new()))
    } else if let Some(s) = first_prefix("launch_") {
        (Some(format!("./{s}")), Some(String::new()))
    } else {
        (None, None)
    };
    (start, stop, port, model)
}

// --- drawing -----------------------------------------------------------------

const KEYBAR: &str = "s start · x stop · l logs · a add · d del · c drop-caches · r refresh · q/Esc quit";

fn draw(f: &mut Frame, app: &AppState, ui: &UiState) {
    let area = f.area();
    let chunks = Layout::vertical([
        Constraint::Length(3), // top bar
        Constraint::Length(9), // dual-node panel
        Constraint::Min(6),    // active endpoints
        Constraint::Length(9), // registry table
        Constraint::Length(1), // footer
    ])
    .split(area);

    draw_top_bar(f, app, chunks[0]);
    draw_nodes(f, app, chunks[1]);
    draw_endpoints(f, app, chunks[2]);
    draw_table(f, app, ui, chunks[3]);
    draw_footer(f, ui, app, chunks[4]);

    if let Some((_, lines)) = &ui.log_view {
        draw_log_drawer(f, lines, bottom_half(chunks[2]));
    }
    if let Some(c) = &ui.confirm {
        draw_confirm(f, c, area);
    }
    if let Some(m) = &ui.add_modal {
        draw_add_modal(f, m, area);
    }
}

fn draw_top_bar(f: &mut Frame, app: &AppState, area: ratatui::layout::Rect) {
    let st = &app.storage;
    let badge = if st.online {
        let used = st.used_bytes.unwrap_or(0);
        let total = st.total_bytes.unwrap_or(0);
        let avail = st.avail_bytes.unwrap_or(0);
        Span::styled(
            format!(
                " [ONLINE: {}G / {}T ({}G avail)]",
                used / 1_073_741_824,
                total / 1_099_511_627_776,
                avail / 1_073_741_824
            ),
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            format!(" [OFFLINE: {}]", st.reason.as_deref().unwrap_or("checks failed")),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    };
    let lines = vec![
        Line::from(vec![
            Span::styled(
                "SPARK-DECK // GB10 CLUSTER CONTROL",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            badge,
        ]),
        Line::from(Span::styled(
            if st.online {
                "GFS2 gate passed on ai + ai2 — start/remove allowed"
            } else {
                "GFS2 gate FAILED — `s` (start) and `a` (add) are blocked"
            },
            Style::default().fg(if st.online { Color::Green } else { Color::Red }),
        )),
        Line::from(Span::styled(
            format!("last update: {}", app.timestamp),
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(Paragraph::new(lines), area);
}

fn mem_gauge_label(node: &Option<NodeState>) -> (String, f64, Style) {
    match node {
        Some(n) => {
            let total = n.mem_total_bytes.unwrap_or(0);
            let used = n.mem_used_bytes.unwrap_or(0);
            let avail = n.mem_available_bytes.unwrap_or(0);
            let cache = n.mem_cache_bytes.unwrap_or(0);
            let swap = n.swap_used_bytes.unwrap_or(0);
            let ratio = if total > 0 {
                (used as f64 / total as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let style = if ratio > 0.9 {
                Style::default().fg(Color::Red)
            } else if ratio > 0.8 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            };
            (
                format!(
                    "Used {} / {} | Avail {} | Cache {} | Swap {}",
                    gib(used),
                    gib(total),
                    gib(avail),
                    gib(cache),
                    gib(swap)
                ),
                ratio,
                style,
            )
        }
        None => ("unreachable (ssh)".into(), 0.0, Style::default().fg(Color::DarkGray)),
    }
}

fn draw_node_half(f: &mut Frame, name: &str, node: &Option<NodeState>, area: ratatui::layout::Rect) {
    let block = Block::default().borders(Borders::ALL).title(name.to_string());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)]).split(inner);
    let (label, ratio, style) = mem_gauge_label(node);
    f.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label(label.clone())
            .gauge_style(style),
        rows[0],
    );
    let disk = node.as_ref().and_then(|n| n.disk_root.as_ref());
    let (dlabel, dratio, dstyle) = match disk {
        Some(d) => {
            let r = if d.total_bytes > 0 {
                (d.used_bytes as f64 / d.total_bytes as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let color = if d.use_pct > 90.0 {
                Color::Red
            } else if d.use_pct > 80.0 {
                Color::Yellow
            } else {
                Color::Green
            };
            (
                format!(
                    "/ {} used | {} avail | {}%",
                    gib(d.used_bytes),
                    gib(d.avail_bytes),
                    d.use_pct as u64
                ),
                r,
                Style::default().fg(color),
            )
        }
        None => ("/ unavailable".into(), 0.0, Style::default().fg(Color::DarkGray)),
    };
    f.render_widget(
        Gauge::default()
            .ratio(dratio)
            .label(dlabel.clone())
            .gauge_style(dstyle),
        rows[2],
    );
    f.render_widget(Paragraph::new(label).style(Style::default()), rows[1]);
    f.render_widget(Paragraph::new(dlabel), rows[3]);
}

fn draw_nodes(f: &mut Frame, app: &AppState, area: ratatui::layout::Rect) {
    let halves = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);
    let ai = app.nodes.get("ai").cloned().flatten();
    let ai2 = app.nodes.get("ai2").cloned().flatten();
    draw_node_half(f, "ai (head · 192.168.20.63)", &ai, halves[0]);
    draw_node_half(f, "ai2 (worker · 192.168.20.229)", &ai2, halves[1]);
}

fn draw_endpoints(f: &mut Frame, app: &AppState, area: ratatui::layout::Rect) {
    let mut lines: Vec<Line> = Vec::new();
    let active: Vec<&EntrySnapshot> = app
        .entries
        .iter()
        .filter(|e| e.state != "STOPPED")
        .collect();
    if active.is_empty() {
        lines.push(Line::from(Span::styled(
            "no active endpoints",
            Style::default().fg(Color::DarkGray),
        )));
    }
    for e in active {
        let (dot, color) = match e.state.as_str() {
            "RUNNING" => ("●", Color::Green),
            "STARTING" | "STOPPING" => ("◐", Color::Yellow),
            _ => ("○", Color::DarkGray),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{dot} "), Style::default().fg(color)),
            Span::styled(&e.id, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" ({}) — {}:{}", e.engine, e.host, e.port)),
            Span::styled(format!(" [{}]", e.state), Style::default().fg(color)),
        ]));
        let mut parts = vec![format!("mem {} ", gib(e.mem_total_bytes))];
        parts.push(format!(
            "(ai: {} | ai2: {}) · ",
            gib(e.mem_ai_bytes),
            gib(e.mem_ai2_bytes)
        ));
        if let Some(c) = e.concurrent_requests {
            let q = e.queued_requests.unwrap_or(0);
            parts.push(format!("conc {c} (+{q} q) · "));
        }
        if let Some(l) = e.live_tok_per_sec {
            parts.push(format!("{l:.1} tok/s (live) · "));
        }
        if let Some(a) = e.lifetime_decode_tok_per_sec {
            parts.push(format!("{a:.1} tok/s (decode avg) · "));
        }
        if let Some(t) = e.total_tokens_generated {
            let mut tail = format!("tot {}", compact(t));
            if let Some(p) = e.prompt_tokens_total {
                tail.push_str(&format!(" (prompt {})", compact(p)));
            }
            if let Some(c) = e.cached_tokens_total {
                tail.push_str(&format!(" (cache {})", compact(c)));
            }
            parts.push(tail);
        }
        if let Some(m) = &e.served_model {
            parts.push(format!(" · model {m}"));
        }
        lines.push(Line::from(Span::raw(parts.concat())));
        if let Some(d) = &e.status_detail {
            lines.push(Line::from(Span::styled(
                format!("  ⚠ {d}"),
                Style::default().fg(Color::Yellow),
            )));
        }
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("ACTIVE ENDPOINTS"))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_table(f: &mut Frame, app: &AppState, ui: &UiState, area: ratatui::layout::Rect) {
    let header = Row::new(["Status", "Name", "Slot", "Port", "Mem", "Path"]).style(
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    );
    let rows = app.entries.iter().enumerate().map(|(i, e)| {
        let (dot, color) = match e.state.as_str() {
            "RUNNING" => ("● RUNNING", Color::Green),
            "STARTING" | "STOPPING" => ("◐ STARTING", Color::Yellow),
            _ => ("○ STOPPED", Color::DarkGray),
        };
        let selected = i == ui.selected;
        let mut style = Style::default();
        if selected {
            style = style.add_modifier(Modifier::REVERSED);
        }
        Row::new(vec![
            Cell::from(dot).style(Style::default().fg(color).patch(style)),
            Cell::from(e.id.clone()).style(style),
            Cell::from(e.slot.clone()).style(style),
            Cell::from(e.port.to_string()).style(style),
            Cell::from(gib(e.mem_total_bytes)).style(style),
            Cell::from(e.path.clone()).style(style),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Min(24),
            Constraint::Length(14),
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("REGISTERED ENTRIES"));
    let mut ts = ratatui::widgets::TableState::default();
    ts.select(Some(ui.selected));
    f.render_stateful_widget(table, area, &mut ts);
}

fn draw_footer(f: &mut Frame, ui: &UiState, app: &AppState, area: ratatui::layout::Rect) {
    let mut spans = vec![Span::styled(KEYBAR, Style::default().fg(Color::DarkGray))];
    if let Some((m, at)) = &ui.message {
        if at.elapsed() < Duration::from_secs(8) {
            spans.push(Span::styled(
                format!("  ⟶ {m}"),
                Style::default()
                    .fg(if m.contains("fail") || m.contains("blocked") { Color::Red } else { Color::Green })
                    .add_modifier(Modifier::BOLD),
            ));
        }
    }
    if !app.storage.online {
        spans.push(Span::styled("  [GFS2 OFFLINE]", Style::default().fg(Color::Red)));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn bottom_half(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let h = area.height / 2;
    ratatui::layout::Rect {
        x: area.x,
        y: area.y + area.height - h,
        width: area.width,
        height: h,
    }
}

fn draw_log_drawer(f: &mut Frame, lines: &[String], area: ratatui::layout::Rect) {
    f.render_widget(Clear, area);
    let text: Vec<Line> = lines.iter().map(|l| Line::from(l.clone())).collect();
    f.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("LOG TAIL (press any key to close)")
                    .style(Style::default().bg(Color::Black)),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_confirm(f: &mut Frame, c: &Confirm, area: ratatui::layout::Rect) {
    let (text, title) = match c {
        Confirm::Stop(id) => (format!("stop `{id}`? (y/n)"), "CONFIRM STOP"),
        Confirm::Remove(id) => (
            format!("remove `{id}` from registry? (y/n) — files are NOT deleted"),
            "CONFIRM REMOVE",
        ),
    };
    let w = 60.min(area.width);
    let h = 5;
    let r = centered(area, w, h);
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(vec![Line::from(text)])
            .block(Block::default().borders(Borders::ALL).title(title))
            .alignment(Alignment::Center),
        r,
    );
}

fn draw_add_modal(f: &mut Frame, m: &AddModal, area: ratatui::layout::Rect) {
    let r = centered(area, 70, 12);
    f.render_widget(Clear, r);
    let labels = ["Path", "ID", "Port", "Slot", "Engine"];
    let values = [&m.path, &m.id, &m.port, &m.slot, &m.engine];
    let mut lines: Vec<Line> = Vec::new();
    for (i, (l, v)) in labels.iter().zip(values.iter()).enumerate() {
        let focused = i == m.focus;
        lines.push(Line::from(vec![
            Span::styled(format!("{l:>6}: "), Style::default().fg(Color::Cyan)),
            if focused {
                Span::styled(format!("{v}█"), Style::default().bg(Color::DarkGray))
            } else {
                Span::raw((*v).clone())
            },
        ]));
    }
    if let Some(n) = &m.note {
        lines.push(Line::from(Span::styled(
            format!("note: {n}"),
            Style::default().fg(Color::Yellow),
        )));
    }
    lines.push(Line::from(Span::styled(
        "Tab next field (auto-detects scripts/.env from Path) · Enter save · Esc cancel",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title("ADD REPOSITORY ENTRY"),
        ),
        r,
    );
}

fn centered(area: ratatui::layout::Rect, w: u16, h: u16) -> ratatui::layout::Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    ratatui::layout::Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

// --- key handling -------------------------------------------------------------

fn handle_key(
    key: crossterm::event::KeyEvent,
    ui: &mut UiState,
    app: &AppState,
    action_tx: &mpsc::UnboundedSender<Action>,
) {
    if key.kind != KeyEventKind::Press {
        return;
    }
    // Log drawer open: any key closes it.
    if ui.log_view.is_some() {
        ui.log_view = None;
        return;
    }
    // Add modal consumes keys.
    if let Some(m) = ui.add_modal.as_mut() {
        match key.code {
            KeyCode::Esc => ui.add_modal = None,
            KeyCode::Tab => {
                if m.focus == 0 {
                    // leaving the path field: try auto-detection
                    let (start, stop, port, model) = detect_launch(&m.path);
                    if let Some(s) = start {
                        m.note = Some(format!("detected start `{s}`{}; port={:?} model={:?}", stop.map(|_| "").unwrap_or_default(), port, model));
                        m.detected_start = Some(s);
                    } else {
                        m.note = Some("no launch.sh/start-*.sh/launch_*.sh found (paths are remote dirs on `ai`)".into());
                        m.detected_start = None;
                    }
                    if let Some(p) = port {
                        m.port = p;
                    }
                    let _ = model;
                }
                m.focus = (m.focus + 1) % 5;
            }
            KeyCode::Enter => {
                if m.focus == 0 {
                    let (start, _stop, port, model) = detect_launch(&m.path);
                    if let Some(s) = start {
                        m.detected_start = Some(s);
                        m.note = Some("detected launch script".into());
                    } else {
                        m.note = Some("no launch script detected; entry will have no start command".into());
                    }
                    if let Some(p) = port {
                        m.port = p;
                    }
                    let _ = model;
                } else {
                    let entry = m.build();
                    ui.add_modal = None;
                    let _ = action_tx.send(Action::Add(entry));
                }
            }
            KeyCode::Backspace => {
                let idx = m.focus;
                m.fields()[idx].pop();
            }
            KeyCode::Char(c) => {
                let idx = m.focus;
                m.fields()[idx].push(c);
            }
            _ => {}
        }
        return;
    }
    // Confirm dialog consumes keys.
    if let Some(c) = ui.confirm.take() {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => match c {
                Confirm::Stop(id) => {
                    let _ = action_tx.send(Action::Stop(id));
                }
                Confirm::Remove(id) => {
                    let _ = action_tx.send(Action::Remove(id));
                }
            },
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {}
            _ => {
                // restore if unhandled key
                ui.confirm = Some(c);
            }
        }
        return;
    }

    let n = app.entries.len();
    match key.code {
        KeyCode::Char('q') => ui.should_quit = true,
        KeyCode::Down | KeyCode::Char('j') => {
            if n > 0 {
                ui.selected = (ui.selected + 1).min(n - 1);
            }
        }
        KeyCode::Up | KeyCode::Char('k') => ui.selected = ui.selected.saturating_sub(1),
        KeyCode::Char('r') => {
            // force refresh: handled by setting a flag via message; collector ticks are short anyway
            ui.message = Some(("refresh requested (collectors run every 2-3s)".into(), Instant::now()));
        }
        KeyCode::Char('s') => {
            if let Some(e) = app.entries.get(ui.selected) {
                if !app.storage.online {
                    ui.message = Some((
                        "start blocked: GFS2 storage offline".into(),
                        Instant::now(),
                    ));
                } else if e.slot == "primary-llm"
                    && app.entries.iter().any(|o| {
                        o.id != e.id && o.slot == "primary-llm" && o.state == "RUNNING"
                    })
                {
                    ui.message = Some((
                        "start blocked: another primary-llm entry is already RUNNING".into(),
                        Instant::now(),
                    ));
                } else {
                    let _ = action_tx.send(Action::Start(e.id.clone()));
                }
            }
        }
        KeyCode::Char('x') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.confirm = Some(Confirm::Stop(e.id.clone()));
            }
        }
        KeyCode::Char('d') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.confirm = Some(Confirm::Remove(e.id.clone()));
            }
        }
        KeyCode::Char('a') => {
            if !app.storage.online {
                ui.message = Some(("add blocked: GFS2 storage offline".into(), Instant::now()));
            } else {
                ui.add_modal = Some(AddModal::new());
            }
        }
        KeyCode::Char('c') => {
            let _ = action_tx.send(Action::DropCaches);
            ui.message = Some(("drop caches requested…".into(), Instant::now()));
        }
        KeyCode::Char('l') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.log_view = Some((e.id.clone(), vec!["loading…".into()]));
                let _ = action_tx.send(Action::ShowLog(e.id.clone()));
            }
        }
        KeyCode::Esc => ui.should_quit = true,
        _ => {}
    }
}

// --- terminal lifecycle --------------------------------------------------------

struct TermGuard;
impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    }
}

async fn run_tui() {
    let client = build_client();
    let registry = load_registry();
    let (tx, rx) = watch::channel(Arc::new(AppState::default()));
    let (action_tx, action_rx) = mpsc::unbounded_channel::<Action>();
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiMsg>();

    let storage = StorageState::default();
    let nodes: BTreeMap<String, Option<NodeState>> = BTreeMap::new();
    let mem: HashMap<String, (u64, u64)> = HashMap::new();
    let probes: HashMap<String, ProbeResult> = HashMap::new();
    let states: HashMap<String, String> = registry
        .iter()
        .map(|e| (e.id.clone(), "STOPPED".to_string()))
        .collect();
    let shared = Arc::new(Shared {
        tx,
        registry: RwLock::new(registry),
        storage: RwLock::new(storage),
        nodes: RwLock::new(nodes),
        mem: RwLock::new(mem),
        probes: RwLock::new(probes),
        states: RwLock::new(states),
        grace: Mutex::new(HashMap::new()),
    });
    // publish the initial snapshot so the first frame has data
    shared.publish();

    tokio::spawn(http_loop(shared.clone(), client));
    tokio::spawn(sys_loop(shared.clone()));
    tokio::spawn(action_loop(action_rx, shared.clone(), ui_tx.clone()));

    let _guard = TermGuard;
    enable_raw_mode().expect("enable_raw_mode");
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("enter alternate screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("terminal");

    let mut ui = UiState::default();
    loop {
        let app = rx.borrow().clone();
        // apply selection bound
        if !app.entries.is_empty() {
            ui.selected = ui.selected.min(app.entries.len() - 1);
        } else {
            ui.selected = 0;
        }
        terminal
            .draw(|f| draw(f, &app, &ui))
            .expect("terminal draw");

        // drain UI messages
        while let Ok(m) = ui_rx.try_recv() {
            match m {
                UiMsg::Message(text) => ui.message = Some((text, Instant::now())),
                UiMsg::Log { id, lines } => ui.log_view = Some((id, lines)),
            }
        }

        if event::poll(Duration::from_millis(250)).expect("event poll") {
            if let Ok(Event::Key(key)) = event::read() {
                if ui.should_quit {
                    break;
                }
                handle_key(key, &mut ui, &app, &action_tx);
            }
        }
        if ui.should_quit {
            break;
        }
    }
    // TermGuard restores raw mode + screen.
}

// =============================================================================
// 9. MAIN + DUMP STATE
// =============================================================================

async fn run_dump() {
    let client = build_client();
    let registry = load_registry();
    let mut cs = CollectorState::default();

    let storage = collect_storage().await;
    let nodes = collect_nodes().await;
    let mem = collect_mem_attr(&registry).await;
    let probes = probe_all(&registry, &mut cs.prev, &client).await;

    let mut states: HashMap<String, String> = HashMap::new();
    for e in &registry {
        let ok = probes.get(&e.id).map(|p| p.ok).unwrap_or(false);
        states.insert(e.id.clone(), if ok { "RUNNING" } else { "STOPPED" }.to_string());
    }

    let app = assemble(&registry, &storage, &nodes, &mem, &probes, &states);
    println!(
        "{}",
        serde_json::to_string_pretty(&app).unwrap_or_else(|_| "{}".to_string())
    );
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "--dump-state") {
        run_dump().await;
        return;
    }
    run_tui().await;
}
