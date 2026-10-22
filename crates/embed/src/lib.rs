//! Shared helpers for the chronos embedding/reranking HTTP servers.
//!
//! Both binaries (`chronos-embed`, `chronos-rerank`) are thin
//! OpenAI/Jina-compatible shims in front of fastembed ONNX models, so they
//! share config parsing and error shaping.

use axum::{http::StatusCode, Json};
use serde_json::{json, Value};

/// Hard cap on request bodies. axum defaults to 2 MB, which a 256-item batch
/// of long chunks would exceed.
pub const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;

/// Error returned by handlers: HTTP status plus an OpenAI-shaped JSON body.
pub type ApiError = (StatusCode, Json<Value>);

pub fn env_or(name: &str, default: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v,
        _ => default.to_string(),
    }
}

pub fn env_num<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + std::fmt::Display,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => match v.trim().parse::<T>() {
            Ok(n) => n,
            Err(e) => {
                eprintln!("[chronos] {name}=\"{v}\" invalid ({e}); using {default}");
                default
            }
        },
        Err(_) => default,
    }
}

pub fn invalid_request(message: impl Into<String>) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": { "message": message.into(), "type": "invalid_request_error" }
        })),
    )
}

pub fn server_error(message: impl Into<String>) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": { "message": message.into(), "type": "server_error" }
        })),
    )
}
