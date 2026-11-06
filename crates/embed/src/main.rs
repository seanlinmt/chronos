//! chronos-embed: OpenAI-compatible BGE-M3 embedding server.
//!
//! Serves `BAAI/bge-m3` (fastembed / ONNX int8, `gpahal/bge-m3-onnx-int8`)
//! over the same `/v1/models` + `/v1/embeddings` surface vLLM exposed, so the
//! chronos registry entry on port 8891 keeps probing RUNNING unchanged.
//!
//! Dense vectors only (1024-d). Sparse/ColBERT outputs are computed by
//! fastembed but discarded here.
//!
//! Config via env (see `Config::from_env`). Runs under systemd; stdout goes to
//! the journal.

use std::sync::{Arc, Mutex};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    routing::{get, post},
    Json, Router,
};
use chronos_embed::{env_num, env_or, invalid_request, server_error, ApiError, MAX_BODY_BYTES};
use fastembed::{Bgem3Embedding, Bgem3InitOptions, Bgem3Model};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

/// Hard cap on texts per request.
const MAX_BATCH: usize = 256;
/// Internal ONNX batch size. Small enough to keep per-request latency sane on
/// GB10 CPU cores without thrashing memory.
const EMBED_BATCH_SIZE: usize = 8;

const DEFAULT_PORT: u16 = 8891;
const DEFAULT_MODEL_CACHE: &str = "/mnt/nvmeof/fastembed-cache";
const DEFAULT_MAX_LEN: usize = 8192;
const DEFAULT_THREADS: usize = 8;
const DEFAULT_MODEL_NAME: &str = "bge-m3";

/// Max ONNX calls executing at once (see `AppState::inflight`). The model
/// mutex already serializes inference, so 1 is the whole point: queued
/// requests park on the semaphore instead of holding blocking threads.
const INFLIGHT_PERMITS: usize = 1;
/// Default tokio pool sizes, overridable via `TOKIO_WORKER_THREADS` /
/// `TOKIO_MAX_BLOCKING_THREADS` (pinned by the systemd unit).
const DEFAULT_TOKIO_WORKERS: usize = 2;
const DEFAULT_TOKIO_BLOCKING: usize = 4;

struct Config {
    port: u16,
    model_cache: String,
    max_len: usize,
    threads: usize,
    model_name: String,
}

impl Config {
    fn from_env() -> Self {
        let threads = env_num::<usize>("CHRONOS_EMBED_THREADS", DEFAULT_THREADS).max(1);
        let max_len = env_num::<usize>("CHRONOS_EMBED_MAX_LEN", DEFAULT_MAX_LEN).max(1);
        Self {
            port: env_num::<u16>("CHRONOS_EMBED_PORT", DEFAULT_PORT),
            model_cache: env_or("CHRONOS_EMBED_MODEL_CACHE", DEFAULT_MODEL_CACHE),
            max_len,
            threads,
            model_name: env_or("CHRONOS_EMBED_MODEL_NAME", DEFAULT_MODEL_NAME),
        }
    }
}

struct AppState {
    /// `embed` needs `&mut self`, so the model is serialized behind a mutex.
    /// The guard is only ever held inside `spawn_blocking`, never across an
    /// `.await`.
    model: Arc<Mutex<Bgem3Embedding>>,
    /// Admission gate to the blocking pool. Without it every concurrent
    /// request spawned its own `spawn_blocking` thread that then just blocked
    /// on the model mutex, and tokio kept those threads alive after the
    /// burst — this grew the process from 28 to 48 OS threads over 13 days of
    /// uptime. With it, the thread count stays flat: extra requests await the
    /// permit cheaply on the async side.
    inflight: Arc<Semaphore>,
    model_name: String,
}

#[derive(Deserialize)]
struct EmbeddingRequest {
    #[serde(default)]
    input: Option<Value>,
    /// Accepted for OpenAI compatibility; only `float` is supported and any
    /// other value is ignored (dense f32 vectors are always returned).
    #[serde(default)]
    #[allow(dead_code)]
    encoding_format: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "chronos"
        }]
    }))
}

/// Normalize the OpenAI `input` field (string or array of strings) into a
/// non-empty batch of non-empty texts, or a 400.
fn extract_texts(input: Option<Value>) -> Result<Vec<String>, ApiError> {
    let input = input.ok_or_else(|| invalid_request("missing required field: input"))?;
    let raw: Vec<String> = match input {
        Value::String(s) => vec![s],
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.into_iter().enumerate() {
                match item {
                    Value::String(s) => out.push(s),
                    _ => {
                        return Err(invalid_request(format!(
                            "input[{i}] must be a string"
                        )))
                    }
                }
            }
            out
        }
        _ => {
            return Err(invalid_request(
                "input must be a string or an array of strings",
            ))
        }
    };

    if raw.is_empty() {
        return Err(invalid_request("input must not be empty"));
    }
    if raw.len() > MAX_BATCH {
        return Err(invalid_request(format!(
            "input batch too large: {} texts (max {MAX_BATCH})",
            raw.len()
        )));
    }
    if let Some(i) = raw.iter().position(|t| t.trim().is_empty()) {
        return Err(invalid_request(format!("input[{i}] is an empty string")));
    }

    Ok(raw)
}

async fn embeddings(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: EmbeddingRequest = serde_json::from_slice(&body)
        .map_err(|e| invalid_request(format!("invalid JSON body: {e}")))?;
    let texts = extract_texts(req.input)?;
    let model_name = state.model_name.clone();
    let model = state.model.clone();
    let inflight = state.inflight.clone();

    // One ONNX call executing at a time (see `AppState::inflight`).
    let _permit = inflight
        .acquire_owned()
        .await
        .map_err(|_| server_error("embedding semaphore closed"))?;

    let dense = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, String> {
        let _permit = _permit; // held for the duration of the ONNX call
        let mut guard = match model.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                eprintln!("[chronos-embed] warning: model mutex was poisoned; recovering");
                poisoned.into_inner()
            }
        };
        let out = guard
            .embed(&texts, Some(EMBED_BATCH_SIZE))
            .map_err(|e| format!("embedding failed: {e}"))?;
        Ok(out.dense)
    })
    .await;

    let dense = match dense {
        Ok(Ok(d)) => d,
        Ok(Err(e)) => {
            eprintln!("[chronos-embed] {e}");
            return Err(server_error(e));
        }
        Err(e) => {
            let msg = format!("embedding task failed: {e}");
            eprintln!("[chronos-embed] {msg}");
            return Err(server_error(msg));
        }
    };

    let data: Vec<Value> = dense
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| {
            json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding
            })
        })
        .collect();

    Ok(Json(json!({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": { "prompt_tokens": 0, "total_tokens": 0 }
    })))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `#[tokio::main]` defaults to one worker per core (20 on GB10) plus a
    // blocking pool that grows toward 512 threads, each kept alive ~10 s past
    // its last task — that silently added 20+ OS threads during concurrent
    // bursts. Pin both pools: past `max_blocking_threads` blocking tasks
    // *queue* instead of spawning threads, which is exactly the backpressure
    // a mutex-serialized ONNX server wants.
    let workers = env_num::<usize>("TOKIO_WORKER_THREADS", DEFAULT_TOKIO_WORKERS).max(1);
    let blocking = env_num::<usize>("TOKIO_MAX_BLOCKING_THREADS", DEFAULT_TOKIO_BLOCKING).max(1);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(blocking)
        .thread_name("chronos-embed")
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_env();
    println!(
        "[chronos-embed] config: port={} model_name={} cache={} max_len={} threads={}",
        cfg.port, cfg.model_name, cfg.model_cache, cfg.max_len, cfg.threads
    );
    println!(
        "[chronos-embed] loading model gpahal/bge-m3-onnx-int8 (first run downloads ~568 MB) ..."
    );

    // Model init is CPU/IO heavy (download + ONNX session build): keep it off
    // the async runtime threads.
    let cache = cfg.model_cache.clone();
    let max_len = cfg.max_len;
    let threads = cfg.threads;
    let loaded = tokio::task::spawn_blocking(move || {
        let opts = Bgem3InitOptions::new(Bgem3Model::BGEM3Q)
            .with_cache_dir(cache.into())
            .with_show_download_progress(false)
            .with_max_length(max_len)
            .with_intra_threads(threads)
            // KleidiAI int8 kernels can produce wrong results on ARM64;
            // disable them on the GB10 aarch64 host.
            .with_session_config("mlas.disable_kleidiai", "1");
        Bgem3Embedding::try_new(opts).map_err(|e| format!("model init failed: {e}"))
    })
    .await;

    let model = match loaded {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            eprintln!("[chronos-embed] {e}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("[chronos-embed] model init task failed: {e}");
            std::process::exit(1);
        }
    };
    println!("[chronos-embed] model loaded; serving");

    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        inflight: Arc::new(Semaphore::new(INFLIGHT_PERMITS)),
        model_name: cfg.model_name,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(embeddings))
        // axum defaults to a 2 MB body cap, which a 256-text batch of long
        // chunks would exceed; MAX_BODY_BYTES covers the worst realistic case.
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], cfg.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("[chronos-embed] listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
