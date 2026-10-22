//! chronos-http — axum web frontend for the 2-node DGX Spark (GB10) cluster.
//!
//! All domain logic (collectors, probes, registry, lifecycle) lives in
//! chronos-core; this crate only serves the embedded UI, adapts core's
//! snapshot to the exact `/api/state` JSON shape the web UI reads, and maps
//! routes onto core actions.
//!
//! Binds 0.0.0.0:8090 by default; override with `CHRONOS_HTTP_PORT`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Html;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use chronos_core::{Collector, RegistryEntry};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BIND_ADDR: &str = "0.0.0.0:8090";
const REMOTE_NODE: &str = "ai2";
const LOGS_TAIL_LINES: usize = 200;
const CONTAINER_LOG_TAIL_LINES: usize = 100;
const STOPPING_DISPLAY_SECS: u64 = 120; // show STOPPING this long after a stop request

fn bind_addr() -> String {
    match std::env::var("CHRONOS_HTTP_PORT") {
        Ok(p) if p.trim().parse::<u16>().is_ok() => format!("0.0.0.0:{}", p.trim()),
        _ => DEFAULT_BIND_ADDR.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Shared app state: core collector + web-UI-specific runtime bookkeeping
// ---------------------------------------------------------------------------

#[derive(Default)]
struct WebState {
    /// entry id -> unix ts when first observed RUNNING/STARTING (for uptime_s).
    running_since: HashMap<String, u64>,
    /// entry ids with a pending stop (display STOPPING until STOPPED/expired).
    stopping: HashMap<String, Instant>,
}

struct AppState {
    collector: Collector,
    web: Mutex<WebState>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
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

async fn fs_read(path: &str) -> Option<String> {
    tokio::fs::read_to_string(path).await.ok()
}

// ---------------------------------------------------------------------------
// /api/state adapter — maps core's snapshot onto the exact key set the
// embedded web UI reads (extracted from static/index.html).
// ---------------------------------------------------------------------------

fn storage_dto(s: &chronos_core::StorageState) -> Value {
    let use_pct = match (s.used_bytes, s.total_bytes) {
        (Some(u), Some(t)) if t > 0 => Some(u as f64 / t as f64 * 100.0),
        _ => None,
    };
    json!({
        "online": s.online,
        "reason": s.reason,
        "checks": {
            "ai_mount": s.checks.ai_findmnt,
            "ai_raid": s.checks.ai_mdstat,
            "ai_stat": s.checks.ai_stat,
            "ai2_mount": s.checks.ai2_findmnt,
            "ai2_stat": s.checks.ai2_stat,
        },
        "disk": {
            "total_bytes": s.total_bytes.unwrap_or(0),
            "used_bytes": s.used_bytes.unwrap_or(0),
            "avail_bytes": s.avail_bytes.unwrap_or(0),
            "use_pct": use_pct.unwrap_or(0.0),
        },
    })
}

fn node_dto(n: &Option<chronos_core::NodeState>) -> Value {
    match n {
        None => Value::Null,
        Some(n) => json!({
            "mem_total_bytes": n.mem_total_bytes,
            "mem_available_bytes": n.mem_available_bytes,
            "mem_used_bytes": n.mem_used_bytes,
            "mem_cache_bytes": n.mem_cache_bytes,
            "swap_total_bytes": n.swap_total_bytes,
            "swap_used_bytes": n.swap_used_bytes,
            "disk": n.disk_root.as_ref().map(|d| json!({
                "mount": "/",
                "total_bytes": d.total_bytes,
                "used_bytes": d.used_bytes,
                "avail_bytes": d.avail_bytes,
                "use_pct": d.use_pct,
            })),
        }),
    }
}

fn telemetry_dto(e: &chronos_core::EntrySnapshot) -> Value {
    json!({
        "concurrent_requests": e.concurrent_requests,
        "queued_requests": e.queued_requests,
        "live_tok_per_sec": e.live_tok_per_sec,
        "lifetime_decode_tok_per_sec": e.lifetime_decode_tok_per_sec,
        "total_tokens_generated": e.total_tokens_generated,
        "prompt_tokens_total": e.prompt_tokens_total,
        "cached_tokens_total": e.cached_tokens_total,
        "http_requests_total": Value::Null,
    })
}

fn mem_opt(v: u64, stopped: bool) -> Value {
    if v == 0 && stopped {
        Value::Null
    } else {
        json!(v)
    }
}

fn entry_dto(e: &RegistryEntry, snap: Option<&chronos_core::EntrySnapshot>, web: &mut WebState, now: u64) -> Value {
    let raw_state = snap.map(|s| s.state.clone()).unwrap_or_else(|| "STOPPED".into());
    let pending_stop = web
        .stopping
        .get(&e.id)
        .map(|at| at.elapsed() < Duration::from_secs(STOPPING_DISPLAY_SECS))
        .unwrap_or(false);
    let status = if raw_state == "STOPPED" {
        web.stopping.remove(&e.id);
        raw_state.clone()
    } else if pending_stop {
        "STOPPING".to_string()
    } else {
        raw_state.clone()
    };

    // uptime bookkeeping: first observation of RUNNING/STARTING/STOPPING.
    let uptime = if status == "STOPPED" {
        web.running_since.remove(&e.id);
        0
    } else {
        let since = *web.running_since.entry(e.id.clone()).or_insert(now);
        now.saturating_sub(since)
    };

    let stopped = raw_state == "STOPPED";
    let mem_ai = snap.map(|s| s.mem_ai_bytes).unwrap_or(0);
    let mem_ai2 = snap.map(|s| s.mem_ai2_bytes).unwrap_or(0);
    let mem_total = snap.map(|s| s.mem_total_bytes).unwrap_or(0);
    let (mem_ai_v, mem_ai2_v, mem_total_v) = if stopped {
        (Value::Null, Value::Null, Value::Null)
    } else {
        (
            mem_opt(mem_ai, false),
            mem_opt(mem_ai2, false),
            mem_opt(mem_total, false),
        )
    };

    json!({
        "id": e.id,
        "name": e.id,
        "path": e.path,
        "start_cmd": e.start,
        "stop_cmd": e.stop,
        "host": e.host,
        "port": e.port,
        "slot": e.slot,
        "engine": e.engine,
        "containers": e.containers,
        "process_match": e.process_match,
        "status": status,
        "status_detail": snap.and_then(|s| s.status_detail.clone()),
        "uptime_s": uptime,
        "served_model": snap.and_then(|s| s.served_model.clone()),
        "mem_ai_bytes": mem_ai_v,
        "mem_ai2_bytes": mem_ai2_v,
        "mem_total_bytes": mem_total_v,
        "telemetry": snap.map(telemetry_dto),
    })
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
    let snap = state.collector.snapshot();
    let reg = state.collector.registry();
    let now = now_unix();
    let mut web = state.web.lock().unwrap();
    let entries: Vec<Value> = reg
        .iter()
        .map(|e| {
            let s = snap.entries.iter().find(|s| s.id == e.id);
            entry_dto(e, s, &mut web, now)
        })
        .collect();
    Json(json!({
        "ts": now,
        "storage": storage_dto(&snap.storage),
        "nodes": {
            "ai": node_dto(snap.nodes.get("ai").unwrap_or(&None)),
            "ai2": node_dto(snap.nodes.get("ai2").unwrap_or(&None)),
        },
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
    slot: Option<String>,
    #[serde(default = "default_engine")]
    engine: String,
    #[serde(default)]
    containers: Vec<String>,
    #[serde(default)]
    process_match: Option<String>,
}

fn default_host() -> String {
    chronos_core::AI_LAN.into()
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
        "tensorfold" => "tensorfold",
        "velogb10" => "velogb10",
        _ => "vllm",
    }
    .to_string();
    let entry = RegistryEntry {
        id,
        path: input.path.trim().to_string(),
        start: input.start_cmd.trim().to_string(),
        stop: input.stop_cmd.trim().to_string(),
        host: input.host.trim().to_string(),
        port: input.port,
        slot: input.slot.unwrap_or_default().trim().to_string(),
        engine,
        containers: input.containers,
        process_match: input.process_match.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
    };

    if state.collector.find(&entry.id).is_some() {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({ "error": format!("entry '{}' already exists", entry.id) })),
        ));
    }
    let dto = entry_dto(&entry, None, &mut state.web.lock().unwrap(), now_unix());
    state
        .collector
        .add(entry.clone())
        .await
        .map_err(|e| (StatusCode::CONFLICT, Json(json!({ "error": e }))))?;
    Ok(Json(dto))
}

async fn update_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(input): Json<EntryInput>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if input.path.trim().is_empty() {
        return Err(bad_request("path is required"));
    }
    if state.collector.find(&id).is_none() {
        return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))));
    }

    let new_id = if input.name.trim().is_empty() || input.name.trim() == id {
        id.clone()
    } else {
        let s = slugify(&input.name);
        if s.is_empty() {
            id.clone()
        } else {
            s
        }
    };

    if new_id != id && state.collector.find(&new_id).is_some() {
        return Err((
            StatusCode::CONFLICT,
            Json(json!({ "error": format!("entry '{new_id}' already exists") })),
        ));
    }

    let engine = match input.engine.as_str() {
        "tensorfold" => "tensorfold",
        "velogb10" => "velogb10",
        _ => "vllm",
    }
    .to_string();

    let entry = RegistryEntry {
        id: new_id.clone(),
        path: input.path.trim().to_string(),
        start: input.start_cmd.trim().to_string(),
        stop: input.stop_cmd.trim().to_string(),
        host: if input.host.trim().is_empty() {
            default_host()
        } else {
            input.host.trim().to_string()
        },
        port: if input.port == 0 { default_port() } else { input.port },
        slot: input.slot.unwrap_or_default().trim().to_string(),
        engine,
        containers: input.containers,
        process_match: input.process_match.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()),
    };

    if new_id != id {
        let mut web = state.web.lock().unwrap();
        if let Some(since) = web.running_since.remove(&id) {
            web.running_since.insert(new_id.clone(), since);
        }
        if let Some(at) = web.stopping.remove(&id) {
            web.stopping.insert(new_id.clone(), at);
        }
    }

    state
        .collector
        .update(&id, entry.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))))?;

    let snap = state.collector.snapshot();
    let s = snap.entries.iter().find(|s| s.id == entry.id);
    let dto = entry_dto(&entry, s, &mut state.web.lock().unwrap(), now_unix());
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
        while let Some(ent) = rd.next_entry().await.ok().flatten() {
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
    if state.collector.find(&id).is_none() {
        return Err((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))));
    }
    state
        .collector
        .remove(&id)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, Json(json!({ "error": e }))))?;
    Ok(Json(json!({ "ok": true })))
}

async fn start_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let entry = state
        .collector
        .find(&id)
        .ok_or((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))))?;

    // Gate on shared storage health.
    let snap = state.collector.snapshot();
    if !snap.storage.online {
        let msg = format!(
            "Shared storage offline: {}",
            snap.storage.reason.clone().unwrap_or_else(|| "unknown".into())
        );
        return Err((StatusCode::CONFLICT, Json(json!({ "error": msg }))));
    }

    // Enforce a single active primary-llm.
    if entry.slot == "primary-llm" {
        let active = snap
            .entries
            .iter()
            .filter(|o| o.id != entry.id && o.slot == "primary-llm")
            .find(|o| matches!(o.state.as_str(), "RUNNING" | "STARTING"));
        if let Some(o) = active {
            let msg = format!("Stop the active primary LLM first: {}", o.id);
            return Err((StatusCode::CONFLICT, Json(json!({ "error": msg }))));
        }
    }

    if let Err(e) = state.collector.start(&entry.id).await {
        // spawn failure — core already returned the entry to its prior state.
        return Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))));
    }
    Ok(Json(json!({ "ok": true })))
}

async fn stop_entry(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let entry = state
        .collector
        .find(&id)
        .ok_or((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))))?;
    state
        .web
        .lock()
        .unwrap()
        .stopping
        .insert(entry.id.clone(), Instant::now());
    if let Err(e) = state.collector.stop(&entry.id).await {
        eprintln!("[stop] {}: {e}", entry.id);
    }
    Ok(Json(json!({ "ok": true })))
}

async fn entry_logs(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let entry = state
        .collector
        .find(&id)
        .ok_or((StatusCode::NOT_FOUND, Json(json!({ "error": "unknown entry" }))))?;

    // If the entry is up and declares containers, show docker logs for them.
    if !entry.containers.is_empty() {
        let running = state
            .collector
            .snapshot()
            .entries
            .iter()
            .find(|s| s.id == entry.id)
            .map(|s| s.state != "STOPPED")
            .unwrap_or(false);
        if running {
            let node = if chronos_core::is_ai2_host(&entry.host) {
                REMOTE_NODE
            } else {
                "ai"
            };
            let mut lines: Vec<String> = Vec::new();
            for c in &entry.containers {
                lines.push(format!("=== {c} ({node}) ==="));
                match chronos_core::container_log_lines(&entry.host, c, CONTAINER_LOG_TAIL_LINES).await {
                    Ok(mut out) => lines.append(&mut out),
                    Err(e) => lines.push(format!("error: {e}")),
                }
            }
            return Ok(Json(json!({ "source": "docker", "lines": lines })));
        }
    }

    // No running containers — fall back to the entry's own log file.
    match chronos_core::log_file_tail(&entry.id, LOGS_TAIL_LINES).await {
        Ok(lines) => Ok(Json(json!({ "source": "file", "lines": lines }))),
        Err(_) => Ok(Json(json!({ "source": "file", "lines": ["(no log file yet)"] }))),
    }
}

async fn drop_caches() -> Json<Value> {
    let r = chronos_core::drop_caches().await;
    Json(json!({ "ok": r.ok, "ai": r.ai, "ai2": r.ai2 }))
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let collector = Collector::spawn();
    let bind = bind_addr();

    let state = Arc::new(AppState {
        collector,
        web: Mutex::new(WebState::default()),
    });

    let app = Router::new()
        .route("/", get(root))
        .route("/api/state", get(api_state))
        .route("/api/entries", post(add_entry))
        .route("/api/entries/inspect", post(inspect_entry))
        .route("/api/entries/{id}", delete(delete_entry).put(update_entry).post(update_entry))
        .route("/api/entries/{id}/start", post(start_entry))
        .route("/api/entries/{id}/stop", post(stop_entry))
        .route("/api/entries/{id}/logs", get(entry_logs))
        .route("/api/cluster/drop-caches", post(drop_caches))
        .with_state(state);

    match tokio::net::TcpListener::bind(&bind).await {
        Ok(listener) => {
            eprintln!("chronos-http listening on {bind}");
            if let Err(e) = axum::serve(listener, app).await {
                eprintln!("server error: {e}");
            }
        }
        Err(e) => {
            eprintln!("failed to bind {bind}: {e}");
            std::process::exit(1);
        }
    }
}
