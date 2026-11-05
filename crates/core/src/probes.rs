//! Engine probes: tensorfold, vllm, velogb10.
//! Probes verify BOTH port reachable AND model name matches the entry,
//! preventing port collisions (e.g. on shared port 8888).

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
        // Unknown/new engines: treat as OpenAI-compatible (/v1/models) — the
        // safest probe for an engine we have no bespoke implementation for.
        "rust-embed" => probe_vllm(e, &url, prev, client).await,
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

// ---------------------------------------------------------------------------
// Model name matching: verifies that host:port is serving the expected model
// ---------------------------------------------------------------------------

const GENERIC_TOKENS: &[&str] = &[
    "launch", "launcher", "launchers", "start", "stop", "server", "serve",
    "sh", "ai", "ai2", "spark", "dgx", "2x", "4x", "flash", "next",
    "exl3", "tensorfold", "vllm", "velogb10", "home", "sean", "repos",
    "mnt", "nvmeof", "chat", "completions", "model", "models", "test",
    // Engine-adjacent words: too generic to identify a served model, and they
    // would otherwise cause false positives across entries sharing a port.
    // ("embed" does not weaken the bge-m3 entries: their ids tokenize to
    // "embedding"/"bge"/"m3", never the bare "embed".)
    "rust", "embed",
];

pub fn normalize_alphanumeric(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn extract_identifying_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for part in text.split(|c: char| !c.is_ascii_alphanumeric() && c != '.') {
        let p = part.trim().to_ascii_lowercase();
        if p.is_empty() {
            continue;
        }
        if p.contains('.') {
            let collapsed: String = p.chars().filter(|c| *c != '.').collect();
            if collapsed.len() >= 3 && !GENERIC_TOKENS.contains(&collapsed.as_str()) {
                tokens.push(collapsed);
            }
        }
        let clean: String = p.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        if clean.len() >= 3 && !GENERIC_TOKENS.contains(&clean.as_str()) {
            tokens.push(clean);
        }
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

pub fn model_matches_entry(e: &RegistryEntry, served_models: &[String]) -> bool {
    if served_models.is_empty() {
        return false;
    }

    let norm_entry = normalize_alphanumeric(&format!("{} {}", e.id, e.path));
    let entry_tokens = extract_identifying_tokens(&format!("{} {}", e.id, e.path));

    for model_id in served_models {
        let norm_model = normalize_alphanumeric(model_id);
        if norm_model.is_empty() {
            continue;
        }

        // Reranker disambiguation: reranker entries only match reranker models.
        if norm_entry.contains("rerank") != norm_model.contains("rerank") {
            continue;
        }

        // Direct containment on full normalized strings.
        if !norm_entry.is_empty() && (norm_entry.contains(&norm_model) || norm_model.contains(&norm_entry)) {
            return true;
        }

        // Match if any non-generic token from entry is in the normalized model name.
        for t in &entry_tokens {
            if norm_model.contains(t) {
                return true;
            }
        }

        // Match if any non-generic token from the model name is in the entry.
        let model_tokens = extract_identifying_tokens(model_id);
        for mt in &model_tokens {
            if norm_entry.contains(mt) {
                return true;
            }
        }
    }

    false
}

async fn find_api_key(path: &str) -> Option<String> {
    for rel in ["config/prod.env", ".env", ".env.local"] {
        let p = format!("{path}/{rel}");
        if let Ok(content) = tokio::fs::read_to_string(&p).await {
            for line in content.lines() {
                let trimmed = line.trim();
                if let Some(rest) = trimmed.strip_prefix("API_KEY=") {
                    let key = rest.trim().trim_matches('"').trim_matches('\'');
                    if !key.is_empty() {
                        return Some(key.to_string());
                    }
                }
            }
        }
    }
    None
}

async fn fetch_v1_models(
    client: &reqwest::Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>, String> {
    let mut req = client.get(format!("{url}/v1/models"));
    if let Some(k) = api_key {
        req = req.header("Authorization", format!("Bearer {k}"));
    }
    let r = match req.send().await {
        Ok(r) => r,
        Err(e) => return Err(format!("http: {e}")),
    };
    let status = r.status();
    if !status.is_success() {
        return Err(format!("models http {status}"));
    }
    let body: Value = match r.json().await {
        Ok(v) => v,
        Err(e) => return Err(format!("models json: {e}")),
    };
    let list: Vec<String> = body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    Ok(list)
}

// ---------------------------------------------------------------------------
// Engine Probes
// ---------------------------------------------------------------------------

async fn probe_tensorfold(
    e: &RegistryEntry,
    url: &str,
    prev: &mut Prev,
    client: &reqwest::Client,
) -> ProbeResult {
    let now = Instant::now();
    let api_key = find_api_key(&e.path).await;

    // 1. Check served model name on /v1/models (requires auth if configured with API key).
    let models = match fetch_v1_models(client, url, api_key.as_deref()).await {
        Ok(m) => m,
        Err(err) => return fail_probe(err),
    };

    let served = models.first().cloned();
    if !model_matches_entry(e, &models) {
        return fail_probe(format!(
            "serving '{}'",
            served.as_deref().unwrap_or("unknown")
        ));
    }

    // 2. Telemetry and health check from /health.
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
    let ok = v
        .get("ok")
        .and_then(|x| x.as_bool())
        .or_else(|| v.get("status").and_then(|x| x.as_str()).map(|s| s == "ok"))
        .unwrap_or(false);
    if !ok {
        return fail_probe("health ok=false".into());
    }
    let live_obj = v.get("live");

    // 3. Prometheus metrics from /metrics (best-effort; supplies counters if /health lacks them).
    let mut metrics_req = client.get(format!("{url}/metrics"));
    if let Some(k) = api_key.as_deref() {
        metrics_req = metrics_req.header("Authorization", format!("Bearer {k}"));
    }
    let metrics: Option<String> = match metrics_req.send().await {
        Ok(r) if r.status().is_success() => r.text().await.ok(),
        _ => None,
    };

    let prom_total = metrics.as_deref().and_then(|txt| {
        find_prom_metric(
            txt,
            &[
                "tensorfold:generation_tokens_total",
                "tensorfold:completion_tokens_total",
                "tensorfold_generation_tokens_total",
                "tensorfold_completion_tokens_total",
                "vllm:generation_tokens_total",
                "generation_tokens_total",
                "completion_tokens_total",
            ],
        )
    }).map(|f| f as u64);

    let prom_prompt = metrics.as_deref().and_then(|txt| {
        find_prom_metric(
            txt,
            &[
                "tensorfold:prompt_tokens_total",
                "tensorfold_prompt_tokens_total",
                "vllm:prompt_tokens_total",
                "prompt_tokens_total",
            ],
        )
    }).map(|f| f as u64);

    let prom_decode = metrics.as_deref().and_then(|txt| {
        find_prom_metric(
            txt,
            &[
                "tensorfold:request_decode_seconds_sum",
                "tensorfold:request_decode_time_seconds_sum",
                "tensorfold_request_decode_seconds_sum",
                "decode_seconds_total",
            ],
        )
    });

    let total = jnum(&v, "completion_tokens_total")
        .or_else(|| jnum(&v, "generation_tokens_total"))
        .or(prom_total);
    let decode = jf(&v, "decode_seconds_total").or(prom_decode);

    let prom_concurrent = metrics.as_deref().and_then(|txt| {
        find_prom_metric(
            txt,
            &[
                "tensorfold:requests_running",
                "tensorfold:num_requests_running",
                "vllm:num_requests_running",
            ],
        )
    }).map(|f| f as u64);

    let prom_queued = metrics.as_deref().and_then(|txt| {
        find_prom_metric(
            txt,
            &[
                "tensorfold:requests_waiting",
                "tensorfold:num_requests_waiting",
                "vllm:num_requests_waiting",
            ],
        )
    }).map(|f| f as u64);

    let concurrent = jnum(&v, "requests_running")
        .or_else(|| jnum(&v, "inflight"))
        .or_else(|| live_obj.and_then(|l| jnum(l, "connections")))
        .or(prom_concurrent);
    let queued = jnum(&v, "queued")
        .or_else(|| live_obj.and_then(|l| jnum(l, "waiting")))
        .or(prom_queued);

    let live = live_obj
        .and_then(|l| jf(l, "decode_tokens_per_second"))
        .or_else(|| match (prev.total, total, prev.decode_s, decode) {
            (Some(pt), Some(nt), Some(pd), Some(nd)) if nd > pd => {
                Some((nt - pt) as f64 / (nd - pd))
            }
            (Some(pt), Some(nt), _, _) if nt > pt => {
                let dt = prev.at.map(|a| now.duration_since(a).as_secs_f64()).unwrap_or(0.0);
                if dt > 0.0 {
                    Some((nt - pt) as f64 / dt)
                } else {
                    None
                }
            }
            (Some(_), Some(_), _, _) => Some(0.0),
            _ => None,
        });

    let lifetime = match (total, decode) {
        (Some(t), Some(d)) if d > 0.0 => Some(t as f64 / d),
        _ => None,
    };
    *prev = Prev {
        total,
        decode_s: decode,
        at: Some(now),
        steps: prev.steps,
    };
    let served = served.or_else(|| v.get("model").and_then(|x| x.as_str()).map(|s| s.to_string()));
    ProbeResult {
        ok: true,
        detail: None,
        served_model: served,
        concurrent,
        queued,
        live_tok_per_sec: live,
        lifetime_decode: lifetime,
        total_tokens: total,
        prompt_tokens: jnum(&v, "prompt_tokens_total").or(prom_prompt),
        cached_tokens: jnum(&v, "cached_tokens_total"),
    }
}

pub fn parse_prom_metric(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .find_map(|l| {
            let trimmed = l.trim();
            if !trimmed.starts_with(key) {
                return None;
            }
            let rest = &trimmed[key.len()..];
            let after_name = if let Some(stripped) = rest.strip_prefix('{') {
                let close_idx = stripped.find('}')?;
                &stripped[close_idx + 1..]
            } else if rest.is_empty() || rest.chars().next()?.is_whitespace() {
                rest
            } else {
                return None;
            };
            after_name.split_whitespace().next().and_then(|v| v.parse().ok())
        })
}

pub fn find_prom_metric(text: &str, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(v) = parse_prom_metric(text, k) {
            return Some(v);
        }
    }
    None
}

/// Mean of a Prometheus histogram: `{base}_sum / {base}_count`. Returns None
/// when either line is missing, count is not > 0, or the result is not finite.
fn prom_histogram_mean(text: &str, base: &str) -> Option<f64> {
    let sum = parse_prom_metric(text, &format!("{base}_sum"))?;
    let count = parse_prom_metric(text, &format!("{base}_count"))?;
    if count > 0.0 {
        let mean = sum / count;
        if mean.is_finite() {
            Some(mean)
        } else {
            None
        }
    } else {
        None
    }
}

pub async fn probe_vllm(e: &RegistryEntry, url: &str, prev: &mut Prev, client: &reqwest::Client) -> ProbeResult {
    let now = Instant::now();
    // Liveness + served model name.
    let models = match fetch_v1_models(client, url, None).await {
        Ok(m) => m,
        Err(err) => return fail_probe(err),
    };

    let served = models.first().cloned();
    if !model_matches_entry(e, &models) {
        return fail_probe(format!(
            "serving '{}'",
            served.as_deref().unwrap_or("unknown")
        ));
    }

    // Metrics are best-effort (may be disabled); only trust a 2xx body.
    let metrics: Option<String> = match client.get(format!("{url}/metrics")).send().await {
        Ok(r) if r.status().is_success() => r.text().await.ok(),
        _ => None,
    };
    let (concurrent, queued, total, prompt) = match &metrics {
        Some(txt) => {
            let run = find_prom_metric(txt, &["vllm:num_requests_running", "num_requests_running"]).map(|f| f as u64);
            let wait = find_prom_metric(txt, &["vllm:num_requests_waiting", "num_requests_waiting"]).map(|f| f as u64);
            let tot = find_prom_metric(txt, &["vllm:generation_tokens_total", "generation_tokens_total"]).map(|f| f as u64);
            let prm = find_prom_metric(txt, &["vllm:prompt_tokens_total", "prompt_tokens_total"]).map(|f| f as u64);
            let fallback_conc = if txt.contains("http_requests_total") { Some(0) } else { None };
            (run.or(fallback_conc), wait, tot, prm)
        }
        None => (None, None, None, None),
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
        steps: prev.steps,
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
        prompt_tokens: prompt,
        cached_tokens: None,
    }
}

/// Aggregate (whole-server) live generation rate for velogb10, matching the
/// aggregate semantics of the vLLM/tensorfold probes' `generation_tokens_total`
/// deltas — NOT a per-session rate. Priority order:
///
/// 1. `requests_running` gauge absent -> None (unknown).
/// 2. zero running -> Some(0.0): definitively idle. This is what lets
///    `chronos wait-idle` open (idle.rs needs live ≈ 0 with concurrent/queued 0).
/// 3. scheduler-steps delta: `velogb10_scheduler_steps_total` moves every step,
///    so it is the continuously-moving source. Each step emits `tok_per_step`
///    accepted tokens per running lane.
/// 4. `step_ms_p50` present and > 0 -> running * 1000/step_ms * tok_per_step.
/// 5. Last resort: the `generation_tokens_total` delta. That counter counts
///    tokens of FINISHED requests only (per the server's own HELP text), so it
///    sits flat while lanes are generating and cannot drive a live rate — it
///    only fires at completion bursts.
///
/// `tok_per_step` is `yield_sum` when speculation is engaged (MTP/DFlash2 emit
/// multiple tokens per step, so `yield_sum > 0`), otherwise 1.0: with
/// speculation off (`mode:"none"`, where the yield fields are null) a step
/// emits exactly one accepted token per running lane.
fn velogb10_live_rate(
    running: Option<u64>,
    prev_steps: Option<u64>,
    steps: Option<u64>,
    dt_secs: f64,
    step_ms_p50: Option<f64>,
    yield_sum: f64,
    prev_total: Option<u64>,
    total: Option<u64>,
) -> Option<f64> {
    let running = running?;
    if running == 0 {
        return Some(0.0);
    }
    let tok_per_step = if yield_sum > 0.0 { yield_sum } else { 1.0 };
    if let (Some(pt), Some(nt)) = (prev_steps, steps) {
        if nt > pt && dt_secs > 0.0 {
            let steps_per_sec = (nt - pt) as f64 / dt_secs;
            return Some(running as f64 * steps_per_sec * tok_per_step);
        }
    }
    if let Some(s) = step_ms_p50 {
        if s > 0.0 {
            return Some(running as f64 * 1000.0 / s * tok_per_step);
        }
    }
    match (prev_total, total) {
        (Some(pt), Some(nt)) if nt > pt && dt_secs > 0.0 => Some((nt - pt) as f64 / dt_secs),
        _ => None,
    }
}

async fn probe_velogb10(
    e: &RegistryEntry,
    url: &str,
    prev: &mut Prev,
    client: &reqwest::Client,
) -> ProbeResult {
    let now = Instant::now();

    // Check model name on /v1/models if available.
    let models = fetch_v1_models(client, url, None).await.ok().unwrap_or_default();
    let served = models.first().cloned();
    if !models.is_empty() && !model_matches_entry(e, &models) {
        return fail_probe(format!(
            "serving '{}'",
            served.as_deref().unwrap_or("unknown")
        ));
    }

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

    // Metrics are best-effort (may be disabled); only trust a 2xx body. The
    // /health emission counters below read 0 when the server runs
    // mode:"none" (no speculation), so /metrics is the source of truth for
    // concurrency, token counters and decode speed.
    let metrics: Option<String> = match client.get(format!("{url}/metrics")).send().await {
        Ok(r) if r.status().is_success() => r.text().await.ok(),
        _ => None,
    };
    let (prom_concurrent, prom_queued, prom_total, prom_prompt, prom_decode_mean, prom_steps, prom_cached) =
        match &metrics {
            Some(txt) => (
                parse_prom_metric(txt, "velogb10_requests_running").map(|f| f as u64),
                parse_prom_metric(txt, "velogb10_requests_waiting").map(|f| f as u64),
                parse_prom_metric(txt, "velogb10_generation_tokens_total").map(|f| f as u64),
                parse_prom_metric(txt, "velogb10_prompt_tokens_total").map(|f| f as u64),
                prom_histogram_mean(txt, "velogb10_decode_tokens_per_second"),
                parse_prom_metric(txt, "velogb10_scheduler_steps_total").map(|f| f as u64),
                parse_prom_metric(txt, "velogb10_prompt_tokens_cached_total").map(|f| f as u64),
            ),
            None => (None, None, None, None, None, None, None),
        };

    let mtp = &v["telemetry"]["mtp"];
    let df = &v["telemetry"]["dflash2"];
    let mtp_emitted = mtp.get("emitted").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
    let df_emitted = df.get("emitted").and_then(|x| x.as_f64()).unwrap_or(0.0) as u64;
    // Prefer the /metrics counter; fall back to the /health speculation
    // counters only when the metric is absent.
    let total = prom_total.or(Some(mtp_emitted + df_emitted));
    let step = v["telemetry"]["step_ms_p50"].as_f64();
    let yield_sum = mtp.get("yield_tok_per_step").and_then(|x| x.as_f64()).unwrap_or(0.0)
        + df.get("yield_tok_per_step").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let live = velogb10_live_rate(
        prom_concurrent,
        prev.steps,
        prom_steps,
        prev.at.map(|a| now.duration_since(a).as_secs_f64()).unwrap_or(0.0),
        step,
        yield_sum,
        prev.total,
        total,
    );
    *prev = Prev {
        total,
        decode_s: prev.decode_s,
        at: Some(now),
        steps: prom_steps,
    };
    ProbeResult {
        ok: true,
        detail: if metrics.is_some() {
            None
        } else {
            Some("stats disabled; liveness via /health".into())
        },
        served_model: served,
        concurrent: prom_concurrent,
        queued: prom_queued,
        live_tok_per_sec: live,
        lifetime_decode: prom_decode_mean,
        total_tokens: total,
        prompt_tokens: prom_prompt,
        cached_tokens: prom_cached,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RegistryEntry;

    fn make_entry(id: &str, path: &str, port: u16) -> RegistryEntry {
        RegistryEntry {
            id: id.into(),
            path: path.into(),
            start: String::new(),
            stop: String::new(),
            host: "192.168.20.63".into(),
            port,
            slot: "primary-llm".into(),
            engine: "tensorfold".into(),
            containers: vec![],
            process_match: None,
        }
    }

    #[test]
    fn test_deepseek_matches_deepseek_entry() {
        let e = make_entry(
            "DeepSeek-v4.1-Flash-2x-DGX-Spark",
            "/home/sean/repos/DeepSeek-v4.1-Flash-2x-DGX-Spark",
            8888,
        );
        let models = vec![
            "deepseek-v41-flash-exl3".to_string(),
            "DeepSeek-V4.1-Flash-1017".to_string(),
        ];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_deepseek_rejects_glm53_entry() {
        let e = make_entry(
            "glm53-tensorfold-spark",
            "/home/sean/repos/glm53-tensorfold-spark",
            8888,
        );
        let models = vec![
            "deepseek-v41-flash-exl3".to_string(),
            "DeepSeek-V4.1-Flash-1017".to_string(),
        ];
        assert!(!model_matches_entry(&e, &models));
    }

    #[test]
    fn test_glm_matches_glm_entry() {
        let e = make_entry(
            "glm53-tensorfold-spark",
            "/home/sean/repos/glm53-tensorfold-spark",
            8888,
        );
        let models = vec!["GLM-5.3-Flash-EXL3".to_string()];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_glm_rejects_deepseek_entry() {
        let e = make_entry(
            "DeepSeek-v4.1-Flash-2x-DGX-Spark",
            "/home/sean/repos/DeepSeek-v4.1-Flash-2x-DGX-Spark",
            8888,
        );
        let models = vec!["GLM-5.3-Flash-EXL3".to_string()];
        assert!(!model_matches_entry(&e, &models));
    }

    #[test]
    fn test_qwen_matches_velogb10() {
        let e = make_entry(
            "veloGB10 (Qwen3.8-Flash-Next)",
            "/home/sean/repos/veloGB10",
            8888,
        );
        let models = vec!["qwen3.8-flash-next".to_string()];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_bge_m3_matches_embedding() {
        let e = make_entry(
            "launch-embedding-ai.sh (BGE-M3)",
            "/mnt/nvmeof/launchers",
            8891,
        );
        let models = vec!["BAAI/bge-m3".to_string()];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_bge_reranker_matches_reranker() {
        let e = make_entry(
            "launch-reranker-ai2.sh (BGE-Reranker-v2-M3)",
            "/mnt/nvmeof/launchers",
            8892,
        );
        let models = vec!["BAAI/bge-reranker-v2-m3".to_string()];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_bge_m3_rejects_reranker_entry() {
        let e = make_entry(
            "launch-reranker-ai2.sh (BGE-Reranker-v2-M3)",
            "/mnt/nvmeof/launchers",
            8892,
        );
        let models = vec!["BAAI/bge-m3".to_string()];
        assert!(!model_matches_entry(&e, &models));
    }

    #[test]
    fn test_deepseek_1017_matches_entry() {
        let e = make_entry(
            "DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark",
            "/home/sean/repos/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark",
            8888,
        );
        let models = vec![
            "deepseek-v41-flash-exl3".to_string(),
            "DeepSeek-V4.1-Flash-1017".to_string(),
        ];
        assert!(model_matches_entry(&e, &models));
    }

    #[test]
    fn test_deepseek_1017_rejects_glm() {
        let e = make_entry(
            "DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark",
            "/home/sean/repos/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark",
            8888,
        );
        let models = vec!["GLM-5.3-Flash-EXL3".to_string()];
        assert!(!model_matches_entry(&e, &models));
    }

    #[test]
    fn test_parse_prom_metric() {
        let sample = "\
# HELP tensorfold:generation_tokens_total Generated tokens of finished requests.
# TYPE tensorfold:generation_tokens_total counter
tensorfold:generation_tokens_total 18836
# HELP vllm:prompt_tokens_total Number of prompt tokens processed.
vllm:prompt_tokens_total{model_name=\"deepseek\"} 732007.5
tensorfold:generation_tokens_total_extra 99999
";
        assert_eq!(
            parse_prom_metric(sample, "tensorfold:generation_tokens_total"),
            Some(18836.0)
        );
        assert_eq!(
            parse_prom_metric(sample, "vllm:prompt_tokens_total"),
            Some(732007.5)
        );
        assert_eq!(
            parse_prom_metric(sample, "non_existent"),
            None
        );
        assert_eq!(
            find_prom_metric(sample, &["missing", "tensorfold:generation_tokens_total"]),
            Some(18836.0)
        );
    }

    // Fixture derived from the velogb10 server's /metrics (v0.7.8), merging
    // two live captures: the original trimmed body plus the scheduler lines
    // from the 4-lane generation evidence. VERBATIM values: requests_running 2,
    // requests_waiting 0, generation_tokens_total 6467, prompt_tokens_total
    // 742487, histogram sum 2369.661016 / count 41, uptime 1476.9,
    // requests_total{finish="stop"} 36, scheduler_steps_total 41242,
    // scheduler_busy 1. SYNTHESIZED: the two _bucket samples, the
    // velogb10_prompt_tokens_cached_total line (that metric exists on the
    // server but its value was not captured — 524288 is a plausible
    // placeholder), and the # HELP/#TYPE texts (generation/decode use the
    // engine's real wording, the rest are approximate). The _bucket lines and
    // the labelled velogb10_requests_total block are kept on purpose: they
    // must not confuse the prefix matching used for the gauges/counters.
    const VELOGB10_METRICS: &str = "\
# HELP velogb10_requests_running Requests currently holding a lane.
# TYPE velogb10_requests_running gauge
velogb10_requests_running 2
# HELP velogb10_requests_waiting Requests queued for a lane.
# TYPE velogb10_requests_waiting gauge
velogb10_requests_waiting 0
# HELP velogb10_generation_tokens_total Generated tokens of finished requests.
# TYPE velogb10_generation_tokens_total counter
velogb10_generation_tokens_total 6467
# TYPE velogb10_prompt_tokens_cached_total counter
velogb10_prompt_tokens_cached_total 524288
# HELP velogb10_prompt_tokens_total Prompt tokens processed.
# TYPE velogb10_prompt_tokens_total counter
velogb10_prompt_tokens_total 742487
# HELP velogb10_decode_tokens_per_second Per-request generation rate after the first token.
# TYPE velogb10_decode_tokens_per_second histogram
velogb10_decode_tokens_per_second_bucket{le=\"10.0\"} 3
velogb10_decode_tokens_per_second_bucket{le=\"+Inf\"} 41
velogb10_decode_tokens_per_second_sum 2369.661016
velogb10_decode_tokens_per_second_count 41
# HELP velogb10_uptime_seconds Server uptime.
# TYPE velogb10_uptime_seconds gauge
velogb10_uptime_seconds 1476.9
# HELP velogb10_scheduler_steps_total Scheduler steps executed.
# TYPE velogb10_scheduler_steps_total counter
velogb10_scheduler_steps_total 41242
# HELP velogb10_scheduler_busy Scheduler busy flag.
# TYPE velogb10_scheduler_busy gauge
velogb10_scheduler_busy 1
# TYPE velogb10_requests_total counter
velogb10_requests_total{finish=\"stop\"} 36
";

    #[test]
    fn test_velogb10_metric_names_parse() {
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_requests_running"), Some(2.0));
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_requests_waiting"), Some(0.0));
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_generation_tokens_total"), Some(6467.0));
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_prompt_tokens_total"), Some(742487.0));
        assert_eq!(
            parse_prom_metric(VELOGB10_METRICS, "velogb10_decode_tokens_per_second_sum"),
            Some(2369.661016)
        );
        assert_eq!(
            parse_prom_metric(VELOGB10_METRICS, "velogb10_decode_tokens_per_second_count"),
            Some(41.0)
        );
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_uptime_seconds"), Some(1476.9));
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_scheduler_steps_total"), Some(41242.0));
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_scheduler_busy"), Some(1.0));
        assert_eq!(
            parse_prom_metric(VELOGB10_METRICS, "velogb10_prompt_tokens_cached_total"),
            Some(524288.0)
        );
        // Labelled lines parse through their {labels} suffix.
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_requests_total"), Some(36.0));
    }

    #[test]
    fn test_velogb10_prefix_match_does_not_leak() {
        // _sum must not be confused with _count or the _bucket lines, and
        // vice versa: the char after the metric name must be whitespace or
        // a {label} block.
        assert_eq!(
            find_prom_metric(VELOGB10_METRICS, &["velogb10_decode_tokens_per_second_sum"]),
            Some(2369.661016)
        );
        assert_eq!(
            find_prom_metric(VELOGB10_METRICS, &["velogb10_decode_tokens_per_second_count"]),
            Some(41.0)
        );
        // The bare histogram base matches none of its _bucket/_sum/_count lines.
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_decode_tokens_per_second"), None);
        // requests_running is not stolen by the labelled requests_total block.
        assert_eq!(find_prom_metric(VELOGB10_METRICS, &["velogb10_requests_running"]), Some(2.0));
        assert_eq!(find_prom_metric(VELOGB10_METRICS, &["velogb10_requests_waiting"]), Some(0.0));
        // prompt_tokens_cached_total is listed before prompt_tokens_total
        // (alphabetical, as the server emits it): neither name may pick up
        // the other's line.
        assert_eq!(parse_prom_metric(VELOGB10_METRICS, "velogb10_prompt_tokens_total"), Some(742487.0));
        assert_eq!(
            parse_prom_metric(VELOGB10_METRICS, "velogb10_prompt_tokens_cached_total"),
            Some(524288.0)
        );
    }

    #[test]
    fn test_velogb10_histogram_mean_is_decode_speed() {
        // sum/count = mean per-request decode speed (VeloBench final_tok_s).
        let mean = prom_histogram_mean(VELOGB10_METRICS, "velogb10_decode_tokens_per_second")
            .expect("sum and count present");
        assert!((mean - 2369.661016 / 41.0).abs() < 1e-9);
        assert!((mean - 57.79).abs() < 0.01);
    }

    #[test]
    fn test_prom_histogram_mean_guards() {
        // count == 0 -> None.
        assert_eq!(prom_histogram_mean("x_sum 1.0\nx_count 0\n", "x"), None);
        // missing sum or missing count -> None.
        assert_eq!(prom_histogram_mean("x_count 4\n", "x"), None);
        assert_eq!(prom_histogram_mean("x_sum 1.0\n", "x"), None);
        // non-finite mean -> None.
        assert_eq!(prom_histogram_mean("x_sum NaN\nx_count 4\n", "x"), None);
    }

    #[test]
    fn test_velogb10_live_rate_steps_delta_is_aggregate() {
        // Verbatim evidence: 4 lanes held 8 s, steps 41065 -> 41242 over 8.04 s,
        // mode:"none" (yield_sum 0 -> 1 token/lane/step). Aggregate =
        // 4 * 177/8.04 ≈ 88; per-lane ≈ 22 steps/s (step_ms_p50 46 ms -> 21.7).
        let v = velogb10_live_rate(
            Some(4),
            Some(41065),
            Some(41242),
            8.04,
            Some(46.0),
            0.0,
            Some(1000),
            Some(1100),
        )
        .expect("steps delta available");
        assert!((v - 4.0 * 177.0 / 8.04).abs() < 1e-9);
        assert!((v - 88.5).abs() < 1.0);
        assert!((v / 4.0 - 22.1).abs() < 0.5);
        // The steps delta wins over both fallbacks: the step_ms arm would give
        // 86.96 and the finished-counter arm (1100-1000)/8.04 = 12.44, neither
        // matches the expected value above.
    }

    #[test]
    fn test_velogb10_live_rate_zero_running_is_definitively_idle() {
        // wait-idle gate: Some(0) must yield Some(0.0), never None, and win
        // over the steps/step_ms/counter arms.
        assert_eq!(
            velogb10_live_rate(Some(0), Some(41065), Some(41242), 2.0, Some(46.0), 0.0, Some(100), Some(200)),
            Some(0.0)
        );
        assert_eq!(velogb10_live_rate(Some(0), None, None, 2.0, None, 0.0, None, None), Some(0.0));
    }

    #[test]
    fn test_velogb10_live_rate_no_running_gauge_is_unknown() {
        // running None -> None with no fallback (unknown, fail-closed for
        // wait-idle), even though the finished-token counter moved.
        assert_eq!(
            velogb10_live_rate(None, Some(1), Some(2), 2.0, Some(46.0), 0.0, Some(100), Some(200)),
            None
        );
    }

    #[test]
    fn test_velogb10_live_rate_step_ms_fallback() {
        // steps absent, step_ms_p50 46 ms, 4 lanes -> 4 * 1000/46 ≈ 87.0.
        let v = velogb10_live_rate(Some(4), None, None, 2.0, Some(46.0), 0.0, None, None)
            .expect("step_ms arm");
        assert!((v - 4.0 * 1000.0 / 46.0).abs() < 1e-9);
        assert!((v - 87.0).abs() < 0.5);
    }

    #[test]
    fn test_velogb10_live_rate_flat_steps_uses_step_ms_arm() {
        // nt == pt (no step movement this tick) with step_ms present -> the
        // step_ms arm, not None. Also covers dt == 0 with steps present.
        assert_eq!(
            velogb10_live_rate(Some(4), Some(41242), Some(41242), 2.0, Some(46.0), 0.0, None, None),
            Some(4.0 * 1000.0 / 46.0)
        );
        assert_eq!(
            velogb10_live_rate(Some(4), Some(41065), Some(41242), 0.0, Some(46.0), 0.0, None, None),
            Some(4.0 * 1000.0 / 46.0)
        );
    }

    #[test]
    fn test_velogb10_live_rate_speculation_scales_with_yield() {
        // mode:"none" baseline is 1 token/lane/step; MTP/DFlash2 yield_sum
        // scales the rate proportionally (yield 2.0 exactly doubles it).
        let base = velogb10_live_rate(Some(4), Some(41065), Some(41242), 8.04, Some(46.0), 0.0, None, None).unwrap();
        let spec = velogb10_live_rate(Some(4), Some(41065), Some(41242), 8.04, Some(46.0), 2.5, None, None).unwrap();
        assert!((spec - 2.5 * base).abs() < 1e-9);
        let doubled = velogb10_live_rate(Some(4), Some(41065), Some(41242), 8.04, Some(46.0), 2.0, None, None).unwrap();
        assert!((doubled - 2.0 * base).abs() < 1e-9);
        // Same scaling on the step_ms arm.
        let ms = velogb10_live_rate(Some(4), None, None, 2.0, Some(46.0), 0.0, None, None).unwrap();
        let ms_spec = velogb10_live_rate(Some(4), None, None, 2.0, Some(46.0), 2.5, None, None).unwrap();
        assert!((ms_spec - 2.5 * ms).abs() < 1e-9);
    }

    #[test]
    fn test_velogb10_live_rate_first_tick_unknown() {
        // dt 0 (or first tick: prev.steps None) with neither step_ms nor the
        // finished counter -> None.
        assert_eq!(velogb10_live_rate(Some(4), Some(41065), Some(41242), 0.0, None, 0.0, None, None), None);
        assert_eq!(velogb10_live_rate(Some(4), None, Some(41242), 2.0, None, 0.0, None, None), None);
        assert_eq!(velogb10_live_rate(Some(4), None, Some(41242), 2.0, Some(0.0), 0.0, None, None), None);
    }

    #[test]
    fn test_velogb10_live_rate_finished_counter_is_last_resort() {
        // No steps metric and no step_ms: the completion-time counter delta
        // (generation_tokens_total counts FINISHED requests only) still works.
        assert_eq!(
            velogb10_live_rate(Some(4), None, None, 2.0, None, 0.0, Some(1000), Some(1100)),
            Some(50.0)
        );
        // Flat completion counter -> None: this is exactly the mode:"none"
        // mid-generation stall that the steps arm now fixes.
        assert_eq!(
            velogb10_live_rate(Some(4), None, None, 2.0, None, 0.0, Some(1100), Some(1100)),
            None
        );
    }
}
