//! Collectors: storage gatekeeper, node metrics, GB10 per-PID GPU attribution.
//!
//! All ai2 data arrives via ONE combined ssh blob per pass (meminfo, df,
//! storage checks, GPU apps, docker ps, per-PID proc samples).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use serde_json::Value;

use crate::model::{
    AI2_LAN, CHECK_TIMEOUT_SECS, DiskInfo, HTTP_TIMEOUT_MS, NodeState, RegistryEntry,
    SSH_BLOB_TIMEOUT_SECS, StorageChecks, StorageState,
};

// =============================================================================
// PARSING / FORMATTING HELPERS
// =============================================================================

/// df -B1 <path> -> (total, used, avail, use_pct)
pub fn parse_df(output: &str) -> Option<(u64, u64, u64, f64)> {
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

pub fn meminfo_key(s: &str, key: &str) -> Option<u64> {
    s.lines()
        .find(|l| l.starts_with(key))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// /proc/meminfo -> (total_kb, avail_kb, cache_kb, swap_total_kb, swap_used_kb)
pub fn parse_meminfo(s: &str) -> Option<(u64, u64, u64, u64, u64)> {
    let total = meminfo_key(s, "MemTotal:")?;
    let avail = meminfo_key(s, "MemAvailable:").unwrap_or_else(|| meminfo_key(s, "MemFree:").unwrap_or(0));
    let buffers = meminfo_key(s, "Buffers:").unwrap_or(0);
    let cached = meminfo_key(s, "Cached:").unwrap_or(0);
    let st = meminfo_key(s, "SwapTotal:").unwrap_or(0);
    let sf = meminfo_key(s, "SwapFree:").unwrap_or(0);
    Some((total, avail, buffers + cached, st, st.saturating_sub(sf)))
}

/// nvidia-smi csv "pid, mib" -> (pid, bytes)
pub fn parse_gpu_apps(s: &str) -> Vec<(u32, u64)> {
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
pub fn parse_docker_ps(s: &str) -> HashMap<String, String> {
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

pub fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / 1_073_741_824.0)
}

pub fn compact(n: u64) -> String {
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

pub fn jnum(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(|x| x.as_f64()).map(|f| f as u64)
}

pub fn jf(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(|x| x.as_f64())
}

/// Hottest valid /sys/class/thermal thermal_zone*/temp value in °C.
///
/// On GB10 the only thermal_zone entries are the acpitz CPU/SoC zones (nvme/mlx5
/// live in hwmon), so the max of valid zones is the CPU/SoC signal. Bogus
/// sensors (e.g. -273150) are discarded.
pub fn read_cpu_temp_max() -> Option<f64> {
    let dir = std::fs::read_dir("/sys/class/thermal").ok()?;
    let mut best: Option<f64> = None;
    for entry in dir.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("thermal_zone") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(entry.path().join("temp")) else {
            continue;
        };
        let Ok(milli) = raw.trim().parse::<i64>() else {
            continue;
        };
        if milli <= 0 || milli > 200_000 {
            continue;
        }
        let c = milli as f64 / 1000.0;
        best = Some(match best {
            Some(b) if b >= c => b,
            _ => c,
        });
    }
    best
}

/// Max valid millidegree value from a newline-separated `temp` dump.
pub fn parse_temp_max(text: &str) -> Option<f64> {
    let mut best: Option<f64> = None;
    for line in text.lines() {
        let Ok(milli) = line.trim().parse::<i64>() else {
            continue;
        };
        if milli <= 0 || milli > 200_000 {
            continue;
        }
        let c = milli as f64 / 1000.0;
        best = Some(match best {
            Some(b) if b >= c => b,
            _ => c,
        });
    }
    best
}

/// Parse the csv line(s) of
/// `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu` into
/// `(temperature °C, utilization %)`.
///
/// `[N/A]` (or any unparsable / out-of-range value) yields `None` for that
/// component only, so a driver that reports temperature but not utilization
/// still surfaces the temperature (and vice versa). The first line carrying a
/// usable value wins; per-GPU lines are collapsed to the first usable sample.
pub fn parse_gpu_stat(text: &str) -> (Option<f64>, Option<f64>) {
    let mut temp_c: Option<f64> = None;
    let mut load_pct: Option<f64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split(',');
        let t = fields
            .next()
            .map(str::trim)
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|c| *c > 0.0 && *c <= 200.0);
        let u = fields
            .next()
            .map(str::trim)
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|p| (0.0..=100.0).contains(p));
        if temp_c.is_none() {
            temp_c = t;
        }
        if load_pct.is_none() {
            load_pct = u;
        }
        if temp_c.is_some() && load_pct.is_some() {
            break;
        }
    }
    (temp_c, load_pct)
}

/// GPU temperature (°C) and utilization (%) from ONE nvidia-smi invocation.
///
/// Both queries are merged because nvidia-smi costs a few hundred ms and this
/// runs per node per tick — never add a second call. `None` when the command
/// fails or prints nothing usable; each inner value is `None` when the driver
/// reports `[N/A]` for just that field.
pub async fn read_gpu_stat() -> Option<(Option<f64>, Option<f64>)> {
    let out = run_cmd(
        "nvidia-smi",
        &[
            "--query-gpu=temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
    )
    .await?;
    let (temp_c, load_pct) = parse_gpu_stat(&out);
    if temp_c.is_none() && load_pct.is_none() {
        None
    } else {
        Some((temp_c, load_pct))
    }
}

/// Parse the aggregate `cpu ` line of /proc/stat into (busy, total).
/// busy = total - (idle + iowait).
pub fn parse_cpu_stat(line: &str) -> Option<(u64, u64)> {
    let rest = line.trim().strip_prefix("cpu")?;
    // Aggregate line only: next char must be whitespace (not "cpu0").
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let fields: Vec<u64> = rest
        .split_whitespace()
        .map(|f| f.parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if fields.len() < 4 {
        return None;
    }
    let total: u64 = fields.iter().copied().fold(0u64, |a, b| a.saturating_add(b));
    let idle = fields[3];
    let iowait = fields.get(4).copied().unwrap_or(0);
    let busy = total.saturating_sub(idle.saturating_add(iowait));
    Some((busy, total))
}

/// Read /proc/stat and return the aggregate (busy, total) counters.
pub fn read_cpu_stat() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/stat").ok()?;
    s.lines().find_map(|l| {
        if l.starts_with("cpu ") || l.starts_with("cpu\t") {
            parse_cpu_stat(l)
        } else {
            None
        }
    })
}

/// Update the stored CPU sample for `node` and return the busy percentage
/// derived from the delta against the previous sample.
pub fn cpu_load_pct(
    prev: &std::sync::Mutex<BTreeMap<String, (u64, u64)>>,
    node: &str,
    sample: (u64, u64),
) -> Option<f64> {
    let old = {
        let mut guard = prev.lock().unwrap();
        let old = guard.insert(node.to_string(), sample);
        old
    };
    let (busy, total) = sample;
    let (old_busy, old_total) = old?;
    let total_delta = total.saturating_sub(old_total);
    if total_delta == 0 {
        return None;
    }
    let busy_delta = busy.saturating_sub(old_busy);
    Some(100.0 * busy_delta as f64 / total_delta as f64)
}

/// Split a marker-delimited blob ("===NAME===") into sections.
pub fn sections(s: &str) -> HashMap<String, String> {
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
pub struct ProcInfo {
    pub pid: u32,
    pub rss_bytes: u64,
    pub container_hex: Option<String>,
    pub cmdline: String,
}

pub fn sanitize_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect()
}

pub fn is_ai2_host(host: &str) -> bool {
    host == AI2_LAN || host == "ai2"
}

pub fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_millis(HTTP_TIMEOUT_MS))
        .build()
        .expect("failed to build reqwest client")
}

// =============================================================================
// COMMAND HELPERS
// =============================================================================

pub async fn run_cmd(program: &str, args: &[&str]) -> Option<String> {
    match tokio::process::Command::new(program).args(args).output().await {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).to_string()),
        _ => None,
    }
}

pub async fn ssh_ai2(cmd: &str) -> Result<String, String> {
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
pub const AI2_BLOB: &str = "echo '===MEMINFO==='; grep -E '^(MemTotal|MemAvailable|Buffers|Cached|SwapTotal|SwapFree):' /proc/meminfo; \
echo '===DFROOT==='; df -B1 /; \
echo '===FINDMNT==='; findmnt -n -o FSTYPE,OPTIONS /mnt/nvmeof; \
echo '===DFNVMEOF==='; df -B1 /mnt/nvmeof; \
echo '===STAT==='; stat -t /mnt/nvmeof/repos /mnt/nvmeof/huggingface >/dev/null 2>&1 && echo OK || echo FAIL; \
echo '===GPUAPPS==='; nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null; \
echo '===DOCKERPS==='; docker ps --no-trunc --format '{{.ID}}|{{.Names}}' 2>/dev/null; \
echo '===PROCS==='; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do \
rss=$(grep VmRSS /proc/$p/status 2>/dev/null | awk '{print $2}'); cg=$(tail -1 /proc/$p/cgroup 2>/dev/null); \
cm=$(tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200); echo \"PROC $p ${rss:-0} $cg\"; echo \"CMD $cm\"; done; \
echo '===CPUPROCS==='; ps -eo pid=,rss=,args= --no-headers --sort=-rss 2>/dev/null | head -200; \
echo '===CPUTEMP==='; cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null; \
echo '===CPUSTAT==='; head -1 /proc/stat; \
echo '===GPUSTAT==='; nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true; \
echo '===END==='";

// =============================================================================
// STORAGE GATEKEEPER
// =============================================================================

/// Storage gatekeeper pass (ai local + ai2 via ssh), every check timeout-wrapped.
pub async fn collect_storage() -> StorageState {
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

// =============================================================================
// NODE METRICS
// =============================================================================

pub async fn node_from_blob(
    blob: &str,
    cpu_prev: &std::sync::Mutex<BTreeMap<String, (u64, u64)>>,
) -> NodeState {
    let sec = sections(blob);
    let mem = sec.get("MEMINFO").and_then(|s| parse_meminfo(s));
    let disk = sec.get("DFROOT").and_then(|s| parse_df(s));
    let cpu_temp = sec.get("CPUTEMP").and_then(|s| parse_temp_max(s));
    let (gpu_temp, gpu_load) = sec
        .get("GPUSTAT")
        .map(|s| parse_gpu_stat(s))
        .unwrap_or((None, None));
    let load = sec
        .get("CPUSTAT")
        .and_then(|s| s.lines().find_map(|l| {
            if l.starts_with("cpu ") || l.starts_with("cpu\t") {
                parse_cpu_stat(l)
            } else {
                None
            }
        }))
        .and_then(|sample| cpu_load_pct(cpu_prev, "ai2", sample));
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
        cpu_load_pct: load,
        cpu_temp_c: cpu_temp,
        gpu_temp_c: gpu_temp,
        gpu_load_pct: gpu_load,
    }
}

pub async fn collect_nodes(
    cpu_prev: &std::sync::Mutex<BTreeMap<String, (u64, u64)>>,
) -> BTreeMap<String, Option<NodeState>> {
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
    let cpu_temp = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        async { read_cpu_temp_max() },
    )
    .await
    .ok()
    .flatten();
    // One nvidia-smi call for both temperature and utilization per tick.
    let (gpu_temp, gpu_load) = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        read_gpu_stat(),
    )
    .await
    .ok()
    .flatten()
    .unwrap_or((None, None));
    let cpu_sample = tokio::time::timeout(
        Duration::from_secs(CHECK_TIMEOUT_SECS),
        async { read_cpu_stat() },
    )
    .await
    .ok()
    .flatten();
    // Guard dropped inside cpu_load_pct before any await below.
    let load = cpu_sample.and_then(|sample| cpu_load_pct(cpu_prev, "ai", sample));
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
                cpu_load_pct: load,
                cpu_temp_c: cpu_temp,
                gpu_temp_c: gpu_temp,
                gpu_load_pct: gpu_load,
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
            Ok(Ok(s)) => Some(node_from_blob(&s, cpu_prev).await),
            _ => None,
        },
    );

    out
}

// =============================================================================
// GB10 PER-PID GPU ATTRIBUTION
// =============================================================================

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
    let mut attributed: HashSet<u32> = HashSet::new();
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
        attributed.insert(pid);
    }

    // CPU-only services (e.g. an embedding server with no CUDA allocation)
    // never appear in the nvidia-smi compute-apps list, so they are invisible
    // to the loop above. Attribute their RSS in a second pass, restricted to
    // entries that explicitly opt in via `process_match` so entries relying on
    // the id fallback do not sweep unrelated processes.
    for p in &procs {
        if attributed.contains(&p.pid) {
            continue;
        }
        let cname = p
            .container_hex
            .as_ref()
            .and_then(|hex| docker.get(hex))
            .map(|s| s.as_str());
        for e in registry {
            if e.process_match.is_none() {
                continue;
            }
            if entry_matches_pid(e, cname, &p.cmdline) {
                *out.entry(e.id.clone()).or_insert(0) += p.rss_bytes;
                attributed.insert(p.pid);
                break;
            }
        }
    }

    out
}

/// Parse the ===CPUPROCS=== section: `ps -eo pid=,rss=,args=` output.
/// RSS is reported in KiB.
fn parse_cpu_procs(s: &str) -> Vec<ProcInfo> {
    let mut out = Vec::new();
    for line in s.lines() {
        let mut it = line.split_whitespace();
        let Some(pid) = it.next().and_then(|x| x.parse::<u32>().ok()) else {
            continue;
        };
        let Some(rss) = it.next().and_then(|x| x.parse::<u64>().ok()) else {
            continue;
        };
        let cmdline = it.collect::<Vec<_>>().join(" ");
        out.push(ProcInfo {
            pid,
            rss_bytes: rss * 1024,
            container_hex: None,
            cmdline,
        });
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

pub async fn collect_mem_attr(registry: &[RegistryEntry]) -> HashMap<String, (u64, u64)> {
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
    let mut procs_ai: Vec<ProcInfo> = parse_gpu_apps(&gpu_ai)
        .iter()
        .map(|(pid, _)| ProcInfo {
            pid: *pid,
            rss_bytes: proc_rss(*pid).unwrap_or(0),
            container_hex: proc_container(*pid),
            cmdline: proc_cmdline(*pid),
        })
        .collect();

    // CPU-only services never appear in the GPU compute-apps list. For entries
    // that declare an explicit `process_match`, discover their PIDs here so the
    // RSS second pass in `attribute_pids` has something to attribute.
    for e in registry {
        let Some(pm) = e.process_match.as_ref() else {
            continue;
        };
        let found = run_cmd("pgrep", &["-f", pm]).await.unwrap_or_default();
        for line in found.lines() {
            let Ok(pid) = line.trim().parse::<u32>() else {
                continue;
            };
            if procs_ai.iter().any(|p| p.pid == pid) {
                continue;
            }
            procs_ai.push(ProcInfo {
                pid,
                rss_bytes: proc_rss(pid).unwrap_or(0),
                container_hex: proc_container(pid),
                cmdline: proc_cmdline(pid),
            });
        }
    }
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
            let mut procs = sec.get("PROCS").map(|s| parse_remote_procs(s)).unwrap_or_default();
            // CPU-only services never appear in GPUAPPS, so fold in the full
            // process list and let the `process_match` second pass attribute
            // them by RSS.
            if let Some(list) = sec.get("CPUPROCS") {
                for p in parse_cpu_procs(list) {
                    if !procs.iter().any(|q| q.pid == p.pid) {
                        procs.push(p);
                    }
                }
            }
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
