//! Engine probes: tensorfold /health, vllm /v1/models + /metrics (with HTTP
//! status checks BEFORE parsing bodies — a shared host:port may answer 401
//! with a valid JSON error body), velogb10 /health, Prev-counter deltas.

use std::collections::HashMap;
use std::time::Instant;

use serde_json::Value;

use crate::collectors::{jf, jnum};
use crate::model::{Prev, ProbeResult, RegistryEntry};

#[derive(Debug, Default)]
pub struct CollectorState {
    pub prev: HashMap<String, Prev>,
}

pub async fn probe_entry(e: &RegistryEntry, prev: &mut Prev, client: &reqwest::Client) -> ProbeResult {
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
        Ok(r) => {
            let status = r.status();
            if !status.is_success() {
                return fail_probe(format!("health http {status}"));
            }
            match r.json().await {
                Ok(v) => v,
                Err(e) => return fail_probe(format!("health json: {e}")),
            }
        }
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

pub async fn probe_vllm(_e: &RegistryEntry, url: &str, prev: &mut Prev, client: &reqwest::Client) -> ProbeResult {
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
        Ok(r) => {
            let status = r.status();
            if !status.is_success() {
                return fail_probe(format!("health http {status}"));
            }
            match r.json().await {
                Ok(v) => v,
                Err(e) => return fail_probe(format!("health json: {e}")),
            }
        }
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

/// Probe all registry entries concurrently; each gets its own prev cell.
pub async fn probe_all(
    registry: &[RegistryEntry],
    prev: &mut HashMap<String, Prev>,
    client: &reqwest::Client,
) -> HashMap<String, ProbeResult> {
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
