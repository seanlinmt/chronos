//! chronos-rerank: OpenAI/Jina-compatible BGE reranker server.
//!
//! Serves `BAAI/bge-reranker-v2-m3` (fastembed / ONNX fp32, CPU) over the
//! `/v1/models` + `/v1/rerank` surface vLLM exposed on port 8892.
//!
//! fastembed returns raw logits; vLLM's `relevance_score` is a probability, so
//! scores are passed through a sigmoid before being returned.

use std::sync::{Arc, Mutex};

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    routing::{get, post},
    Json, Router,
};
use chronos_embed::{env_num, env_or, invalid_request, server_error, ApiError, MAX_BODY_BYTES};
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use serde::Deserialize;
use serde_json::{json, Value};

/// Hard cap on documents per request.
const MAX_DOCUMENTS: usize = 256;
/// Internal ONNX batch size.
const RERANK_BATCH_SIZE: usize = 8;

const DEFAULT_PORT: u16 = 8892;
const DEFAULT_MODEL_CACHE: &str = "/mnt/nvmeof/fastembed-cache";
const DEFAULT_MAX_LEN: usize = 8192;
const DEFAULT_THREADS: usize = 8;
const DEFAULT_MODEL_NAME: &str = "bge-reranker-v2-m3";

struct Config {
    port: u16,
    model_cache: String,
    max_len: usize,
    threads: usize,
    model_name: String,
}

impl Config {
    fn from_env() -> Self {
        Self {
            port: env_num::<u16>("CHRONOS_RERANK_PORT", DEFAULT_PORT),
            model_cache: env_or("CHRONOS_RERANK_MODEL_CACHE", DEFAULT_MODEL_CACHE),
            max_len: env_num::<usize>("CHRONOS_RERANK_MAX_LEN", DEFAULT_MAX_LEN).max(1),
            threads: env_num::<usize>("CHRONOS_RERANK_THREADS", DEFAULT_THREADS).max(1),
            model_name: env_or("CHRONOS_RERANK_MODEL_NAME", DEFAULT_MODEL_NAME),
        }
    }
}

struct AppState {
    /// `rerank` needs `&mut self`, so the model is serialized behind a mutex.
    model: Arc<Mutex<TextRerank>>,
    model_name: String,
}

#[derive(Deserialize)]
struct RerankRequest {
    #[serde(default)]
    query: Option<Value>,
    #[serde(default)]
    documents: Option<Value>,
    #[serde(default)]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: Option<bool>,
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
}

fn extract_query(query: Option<Value>) -> Result<String, ApiError> {
    match query {
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(s),
        Some(Value::String(_)) | None => Err(invalid_request("missing or empty required field: query")),
        _ => Err(invalid_request("query must be a string")),
    }
}

fn extract_documents(documents: Option<Value>) -> Result<Vec<String>, ApiError> {
    let documents = documents.ok_or_else(|| invalid_request("missing required field: documents"))?;
    let items = match documents {
        Value::Array(items) => items,
        _ => return Err(invalid_request("documents must be an array")),
    };
    if items.is_empty() {
        return Err(invalid_request("documents must not be empty"));
    }
    if items.len() > MAX_DOCUMENTS {
        return Err(invalid_request(format!(
            "too many documents: {} (max {MAX_DOCUMENTS})",
            items.len()
        )));
    }

    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let text = match item {
            Value::String(s) => s,
            Value::Object(map) => match map.get("text") {
                Some(Value::String(s)) => s.clone(),
                _ => {
                    return Err(invalid_request(format!(
                        "documents[{i}].text must be a string"
                    )))
                }
            },
            _ => {
                return Err(invalid_request(format!(
                    "documents[{i}] must be a string or an object with a text field"
                )))
            }
        };
        if text.trim().is_empty() {
            return Err(invalid_request(format!("documents[{i}] is an empty string")));
        }
        out.push(text);
    }
    Ok(out)
}

/// fastembed yields raw logits; map to a 0..1 relevance probability like vLLM.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
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

async fn rerank(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let req: RerankRequest = serde_json::from_slice(&body)
        .map_err(|e| invalid_request(format!("invalid JSON body: {e}")))?;

    let query = extract_query(req.query)?;
    let documents = extract_documents(req.documents)?;
    let return_documents = req.return_documents.unwrap_or(false);
    let top_n = req
        .top_n
        .unwrap_or(documents.len())
        .clamp(1, documents.len());

    let model_name = state.model_name.clone();
    let model = state.model.clone();
    let docs_for_model = documents.clone();

    let scored = tokio::task::spawn_blocking(move || -> Result<Vec<(usize, f32)>, String> {
        let mut guard = match model.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                eprintln!("[chronos-rerank] warning: model mutex was poisoned; recovering");
                poisoned.into_inner()
            }
        };
        let docs: Vec<&str> = docs_for_model.iter().map(|s| s.as_str()).collect();
        let results = guard
            .rerank(query.as_str(), &docs, true, Some(RERANK_BATCH_SIZE))
            .map_err(|e| format!("rerank failed: {e}"))?;
        Ok(results
            .into_iter()
            .map(|r| (r.index, sigmoid(r.score)))
            .collect())
    })
    .await;

    let mut scored = match scored {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            eprintln!("[chronos-rerank] {e}");
            return Err(server_error(e));
        }
        Err(e) => {
            let msg = format!("rerank task failed: {e}");
            eprintln!("[chronos-rerank] {msg}");
            return Err(server_error(msg));
        }
    };

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(top_n);

    let results: Vec<Value> = scored
        .into_iter()
        .map(|(index, score)| {
            let mut obj = json!({ "index": index, "relevance_score": score });
            if return_documents {
                obj["document"] = json!({ "text": documents[index] });
            }
            obj
        })
        .collect();

    Ok(Json(json!({
        "model": model_name,
        "usage": { "prompt_tokens": 0, "total_tokens": 0 },
        "results": results
    })))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_env();
    println!(
        "[chronos-rerank] config: port={} model_name={} cache={} max_len={} threads={}",
        cfg.port, cfg.model_name, cfg.model_cache, cfg.max_len, cfg.threads
    );
    println!(
        "[chronos-rerank] loading model rozgo/bge-reranker-v2-m3 (first run downloads ~2.3 GB) ..."
    );

    let cache = cfg.model_cache.clone();
    let max_len = cfg.max_len;
    let threads = cfg.threads;
    let loaded = tokio::task::spawn_blocking(move || {
        let opts = RerankInitOptions::new(RerankerModel::BGERerankerV2M3)
            .with_cache_dir(cache.into())
            .with_show_download_progress(false)
            .with_max_length(max_len)
            .with_intra_threads(threads)
            // KleidiAI int8 kernels can produce wrong results on ARM64.
            .with_session_config("mlas.disable_kleidiai", "1");
        TextRerank::try_new(opts).map_err(|e| format!("model init failed: {e}"))
    })
    .await;

    let model = match loaded {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => {
            eprintln!("[chronos-rerank] {e}");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("[chronos-rerank] model init task failed: {e}");
            std::process::exit(1);
        }
    };
    println!("[chronos-rerank] model loaded; serving");

    let state = Arc::new(AppState {
        model: Arc::new(Mutex::new(model)),
        model_name: cfg.model_name,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/rerank", post(rerank))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], cfg.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("[chronos-rerank] listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
