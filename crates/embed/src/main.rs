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

    let dense = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>, String> {
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
