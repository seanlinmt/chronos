//! Idle gate — "wait until the box is provably quiet" for agent/cron gating.
//!
//! `concurrent_requests` is a **2 s point gauge** (`HTTP_INTERVAL_SECS`), so a
//! single zero reading proves almost nothing. The gate therefore requires, for
//! *every* sample across the whole window:
//!
//! - `concurrent_requests == 0`
//! - `queued_requests == 0`
//! - `live_tok_per_sec ~= 0` — derived in core from the monotonic
//!   `generation_tokens_total` counter, so a flat window also means no tokens
//!   were generated between samples. This is what closes the "request started
//!   and finished between two polls" hole for decode-heavy work.
//!
//! Fail-closed: a `None` gauge (probe failed, entry stopped/renamed, `/metrics`
//! disabled, CPU-only entry) resets the window instead of counting as idle.
//! `allow_unknown` opts out of that for entries that never report telemetry.
//!
//! Nothing here prints; callers receive [`IdleEvent`]s so the TUI can log to
//! stderr and a future HTTP endpoint can stream them.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use crate::collectors::build_client;
use crate::model::{Prev, ProbeResult, RegistryEntry};
use crate::probes::probe_all;
use crate::registry::load_or_seed_registry;

/// Default idle window required before the gate opens.
pub const DEFAULT_WINDOW_SECS: u64 = 60;
/// Default poll period. Chronos's own loops sample every 2 s; probing a single
/// entry is cheaper, but 2 s keeps the gate's cost identical to one tick.
pub const DEFAULT_SAMPLE_SECS: u64 = 2;
/// Default total budget before the gate gives up and reports `TimedOut`.
pub const DEFAULT_MAX_WAIT_SECS: u64 = 300;
/// Tokens/s at or below this reads as "not decoding".
pub const TOK_EPSILON: f64 = 0.5;

// =============================================================================
// SAMPLE + VERDICT (pure)
// =============================================================================

/// The three gauges that decide idleness, taken from one probe.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sample {
    pub concurrent: Option<u64>,
    pub queued: Option<u64>,
    pub tok_per_sec: Option<f64>,
}

impl Sample {
    pub fn from_probe(p: &ProbeResult) -> Self {
        Sample {
            concurrent: p.concurrent,
            queued: p.queued,
            tok_per_sec: p.live_tok_per_sec,
        }
    }

    /// Classify one probe result. Anything unresolvable is `Unknown`, never
    /// `Idle` — the caller decides whether to fail closed.
    pub fn verdict(&self) -> Verdict {
        match (self.concurrent, self.queued) {
            (Some(0), Some(0)) => {
                if self.tok_per_sec.unwrap_or(0.0) > TOK_EPSILON {
                    Verdict::Busy
                } else {
                    Verdict::Idle
                }
            }
            (None, _) | (_, None) => Verdict::Unknown,
            _ => Verdict::Busy,
        }
    }
}

impl fmt::Display for Sample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn opt<T: fmt::Display>(v: Option<T>) -> String {
            v.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        }
        write!(
            f,
            "running={} queued={} tok/s={}",
            opt(self.concurrent),
            opt(self.queued),
            opt(self.tok_per_sec.map(|t| (t * 10.0).round() / 10.0)),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Idle,
    Busy,
    Unknown,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Idle => "idle",
            Verdict::Busy => "busy",
            Verdict::Unknown => "unknown",
        })
    }
}

// =============================================================================
// TRACKER (pure, unit-tested)
// =============================================================================

/// Result of folding one sample into the tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tick {
    /// Window just started on this sample.
    Started,
    /// Still inside the window, not yet long enough.
    Continued,
    /// A non-idle sample cleared the window.
    Reset,
    /// The window has been satisfied for this long.
    Open(Duration),
}

/// Accumulates consecutive idle samples into "idle for at least `window`".
#[derive(Debug, Clone)]
pub struct IdleTracker {
    window: Duration,
    allow_unknown: bool,
    idle_since: Option<Instant>,
}

impl IdleTracker {
    pub fn new(window: Duration) -> Self {
        IdleTracker {
            window,
            allow_unknown: false,
            idle_since: None,
        }
    }

    /// Treat `Unknown` samples as idle (for entries that never report gauges).
    pub fn allow_unknown(mut self, yes: bool) -> Self {
        self.allow_unknown = yes;
        self
    }

    /// Fold one sample taken at `now`.
    pub fn observe(&mut self, now: Instant, verdict: Verdict) -> Tick {
        let counts_as_idle =
            verdict == Verdict::Idle || (self.allow_unknown && verdict == Verdict::Unknown);
        if !counts_as_idle {
            self.idle_since = None;
            return Tick::Reset;
        }
        match self.idle_since {
            None => {
                self.idle_since = Some(now);
                if self.window.is_zero() {
                    Tick::Open(Duration::ZERO)
                } else {
                    Tick::Started
                }
            }
            Some(since) => {
                let elapsed = now.saturating_duration_since(since);
                if elapsed >= self.window {
                    Tick::Open(elapsed)
                } else {
                    Tick::Continued
                }
            }
        }
    }
}

// =============================================================================
// OPTIONS + OUTCOME
// =============================================================================

#[derive(Debug, Clone)]
pub struct GateOptions {
    /// Registry entry id, or `"auto"` for "the RUNNING entry that reports
    /// telemetry" (survives model swaps without a hardcoded id).
    pub entry: String,
    pub window: Duration,
    pub sample: Duration,
    pub max_wait: Duration,
    pub allow_unknown: bool,
}

impl Default for GateOptions {
    fn default() -> Self {
        GateOptions {
            entry: "auto".into(),
            window: Duration::from_secs(DEFAULT_WINDOW_SECS),
            sample: Duration::from_secs(DEFAULT_SAMPLE_SECS),
            max_wait: Duration::from_secs(DEFAULT_MAX_WAIT_SECS),
            allow_unknown: false,
        }
    }
}

impl GateOptions {
    /// Reject configurations that can never succeed or would spin.
    pub fn validate(&self) -> Result<(), String> {
        if self.sample.is_zero() {
            return Err("sample period must be > 0s".into());
        }
        if self.window.is_zero() {
            return Err("window must be > 0s".into());
        }
        if self.max_wait <= self.window {
            return Err(format!(
                "max-wait ({}s) must be greater than the window ({}s)",
                self.max_wait.as_secs(),
                self.window.as_secs()
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdleOutcome {
    Open {
        entry: String,
        idle_for: Duration,
        waited: Duration,
    },
    TimedOut {
        entry: String,
        waited: Duration,
        last: Sample,
    },
    UnknownEntry {
        entry: String,
        known: Vec<String>,
    },
}

impl IdleOutcome {
    /// Process exit code: 0 open, 1 not idle in time, 2 bad target.
    pub fn exit_code(&self) -> i32 {
        match self {
            IdleOutcome::Open { .. } => 0,
            IdleOutcome::TimedOut { .. } => 1,
            IdleOutcome::UnknownEntry { .. } => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdleEvent {
    Resolved {
        entry: String,
        entries: usize,
    },
    Started {
        entry: String,
        window_secs: u64,
    },
    Reset {
        entry: String,
        verdict: Verdict,
        sample: Sample,
    },
    Open {
        entry: String,
        idle_secs: u64,
    },
    Timeout {
        entry: String,
        waited_secs: u64,
        sample: Sample,
    },
}

impl fmt::Display for IdleEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdleEvent::Resolved { entry, entries } => write!(
                f,
                "target={entry} (selected from {entries} registry entries)"
            ),
            IdleEvent::Started { entry, window_secs } => write!(
                f,
                "{entry} idle — window started, need {window_secs}s without interruption"
            ),
            IdleEvent::Reset {
                entry,
                verdict,
                sample,
            } => write!(f, "{entry} {verdict} ({sample}) — window reset"),
            IdleEvent::Open { entry, idle_secs } => {
                write!(f, "OPEN: {entry} idle for {idle_secs}s — safe to proceed")
            }
            IdleEvent::Timeout {
                entry,
                waited_secs,
                sample,
            } => write!(
                f,
                "TIMEOUT: {entry} never held idle for the full window in {waited_secs}s (last sample: {sample}) — do not proceed"
            ),
        }
    }
}

// =============================================================================
// DRIVER
// =============================================================================

/// Resolve `want` against a probed registry.
///
/// `auto` targets the GPU server, in this order: a healthy `primary-llm` entry
/// that actually reports `concurrent_requests`, any healthy `primary-llm`, any
/// `primary-llm` at all (so a stopped GPU server reads as "gate closed", not as
/// "watching an unrelated embedding service"), then the same ladder over every
/// entry. Slot-first ordering is what keeps a model swap from silently
/// retargeting the gate at `bge-m3`.
pub fn resolve_target(
    want: &str,
    registry: &[RegistryEntry],
    probes: &HashMap<String, ProbeResult>,
) -> Option<RegistryEntry> {
    if !want.is_empty() && want != "auto" {
        return registry.iter().find(|e| e.id == want).cloned();
    }

    fn healthy(probes: &HashMap<String, ProbeResult>, e: &RegistryEntry) -> bool {
        probes.get(&e.id).map(|p| p.ok).unwrap_or(false)
    }
    fn reports(probes: &HashMap<String, ProbeResult>, e: &RegistryEntry) -> bool {
        probes.get(&e.id).map(|p| p.concurrent.is_some()).unwrap_or(false)
    }
    let primary = || registry.iter().filter(|e| e.slot.eq_ignore_ascii_case("primary-llm"));

    primary()
        .find(|e| healthy(probes, e) && reports(probes, e))
        .or_else(|| primary().find(|e| healthy(probes, e)))
        .or_else(|| primary().next())
        .or_else(|| registry.iter().find(|e| healthy(probes, e) && reports(probes, e)))
        .or_else(|| registry.iter().find(|e| healthy(probes, e)))
        .cloned()
}

/// Poll until the target has been idle for `opts.window`, or give up.
///
/// One full registry pass resolves the target (so `auto` can see every entry);
/// every later tick probes only that entry.
pub async fn wait_idle<F>(opts: &GateOptions, mut on_event: F) -> IdleOutcome
where
    F: FnMut(&IdleEvent),
{
    let client = build_client();
    let registry = load_or_seed_registry();
    let mut prev: HashMap<String, Prev> = HashMap::new();

    let all = probe_all(&registry, &mut prev, &client).await;
    let target = match resolve_target(&opts.entry, &registry, &all) {
        Some(t) => t,
        None => {
            return IdleOutcome::UnknownEntry {
                entry: opts.entry.clone(),
                known: registry.iter().map(|e| e.id.clone()).collect(),
            }
        }
    };
    on_event(&IdleEvent::Resolved {
        entry: target.id.clone(),
        entries: registry.len(),
    });

    let mut tracker = IdleTracker::new(opts.window).allow_unknown(opts.allow_unknown);
    let start = Instant::now();

    // The resolve pass above already sampled the target. Wait one poll period
    // before sampling again: back-to-back probes would compute the tok/s delta
    // over ~0ms and could report a spurious burst.
    tokio::time::sleep(opts.sample).await;

    loop {
        let results = probe_all(std::slice::from_ref(&target), &mut prev, &client).await;
        let last = results
            .get(&target.id)
            .map(Sample::from_probe)
            // A missing result means the probe task failed: unresolvable, so
            // `Unknown` and (by default) the window resets.
            .unwrap_or_default();
        let now = Instant::now();

        match tracker.observe(now, last.verdict()) {
            Tick::Started => on_event(&IdleEvent::Started {
                entry: target.id.clone(),
                window_secs: opts.window.as_secs(),
            }),
            Tick::Continued => {}
            Tick::Reset => on_event(&IdleEvent::Reset {
                entry: target.id.clone(),
                verdict: last.verdict(),
                sample: last,
            }),
            Tick::Open(idle_for) => {
                on_event(&IdleEvent::Open {
                    entry: target.id.clone(),
                    idle_secs: idle_for.as_secs(),
                });
                return IdleOutcome::Open {
                    entry: target.id,
                    idle_for,
                    waited: now.saturating_duration_since(start),
                };
            }
        }

        let waited = now.saturating_duration_since(start);
        if waited >= opts.max_wait {
            on_event(&IdleEvent::Timeout {
                entry: target.id.clone(),
                waited_secs: waited.as_secs(),
                sample: last,
            });
            return IdleOutcome::TimedOut {
                entry: target.id,
                waited,
                last,
            };
        }
        tokio::time::sleep(opts.sample).await;
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(concurrent: Option<u64>, queued: Option<u64>, tok: Option<f64>) -> Sample {
        Sample {
            concurrent,
            queued,
            tok_per_sec: tok,
        }
    }

    #[test]
    fn verdict_requires_all_three_gauges_zero() {
        assert_eq!(sample(Some(0), Some(0), Some(0.0)).verdict(), Verdict::Idle);
        assert_eq!(
            sample(Some(0), Some(0), Some(0.4)).verdict(),
            Verdict::Idle,
            "sub-epsilon tok/s is noise, still idle"
        );
        assert_eq!(
            sample(Some(0), Some(0), Some(3.0)).verdict(),
            Verdict::Busy,
            "decoding between samples is not idle"
        );
        assert_eq!(sample(Some(1), Some(0), Some(0.0)).verdict(), Verdict::Busy);
        assert_eq!(
            sample(Some(0), Some(2), Some(0.0)).verdict(),
            Verdict::Busy,
            "a queued request means the box is not free"
        );
        assert_eq!(
            sample(None, Some(0), Some(0.0)).verdict(),
            Verdict::Unknown,
            "fail closed on missing gauges"
        );
        assert_eq!(sample(Some(0), None, None).verdict(), Verdict::Unknown);
        assert_eq!(Sample::default().verdict(), Verdict::Unknown);
    }

    #[test]
    fn tracker_opens_only_after_window_elapses() {
        let t0 = Instant::now();
        let mut tr = IdleTracker::new(Duration::from_secs(60));
        assert_eq!(tr.observe(t0, Verdict::Idle), Tick::Started);
        assert_eq!(tr.observe(t0 + Duration::from_secs(2), Verdict::Idle), Tick::Continued);
        assert_eq!(tr.observe(t0 + Duration::from_secs(58), Verdict::Idle), Tick::Continued);
        assert_eq!(
            tr.observe(t0 + Duration::from_secs(60), Verdict::Idle),
            Tick::Open(Duration::from_secs(60))
        );
    }

    #[test]
    fn one_busy_sample_resets_the_window() {
        let t0 = Instant::now();
        let mut tr = IdleTracker::new(Duration::from_secs(10));
        assert_eq!(tr.observe(t0, Verdict::Idle), Tick::Started);
        assert_eq!(tr.observe(t0 + Duration::from_secs(4), Verdict::Idle), Tick::Continued);
        assert_eq!(tr.observe(t0 + Duration::from_secs(6), Verdict::Busy), Tick::Reset);
        // Window restarts from zero, so 8s of idle after the blip is not enough.
        assert_eq!(tr.observe(t0 + Duration::from_secs(8), Verdict::Idle), Tick::Started);
        assert_eq!(tr.observe(t0 + Duration::from_secs(16), Verdict::Idle), Tick::Continued);
        assert_eq!(
            tr.observe(t0 + Duration::from_secs(18), Verdict::Idle),
            Tick::Open(Duration::from_secs(10))
        );
    }

    #[test]
    fn unknown_resets_unless_allowed() {
        let t0 = Instant::now();
        let mut strict = IdleTracker::new(Duration::from_secs(4));
        assert_eq!(strict.observe(t0, Verdict::Idle), Tick::Started);
        assert_eq!(
            strict.observe(t0 + Duration::from_secs(2), Verdict::Unknown),
            Tick::Reset,
            "a failed probe cannot be counted as idle"
        );

        let mut lax = IdleTracker::new(Duration::from_secs(4)).allow_unknown(true);
        assert_eq!(lax.observe(t0, Verdict::Unknown), Tick::Started);
        assert_eq!(
            lax.observe(t0 + Duration::from_secs(4), Verdict::Unknown),
            Tick::Open(Duration::from_secs(4))
        );
    }

    #[test]
    fn zero_window_opens_on_first_idle_sample() {
        let mut tr = IdleTracker::new(Duration::ZERO);
        assert_eq!(tr.observe(Instant::now(), Verdict::Idle), Tick::Open(Duration::ZERO));
    }

    #[test]
    fn validate_rejects_impossible_configs() {
        let mut o = GateOptions::default();
        assert!(o.validate().is_ok());

        o.sample = Duration::ZERO;
        assert!(o.validate().is_err());
        o = GateOptions::default();
        o.window = Duration::ZERO;
        assert!(o.validate().is_err());
        o = GateOptions::default();
        o.max_wait = o.window;
        assert!(o.validate().is_err(), "max-wait must exceed the window");
    }

    #[test]
    fn auto_prefers_healthy_entry_with_telemetry() {
        let reg: Vec<RegistryEntry> = serde_json::from_str(
            r#"[
                {"id":"stopped-llm","path":"/x","start":"","stop":"","port":8888,"slot":"primary-llm","engine":"vllm","host":"192.168.20.63","containers":[],"process_match":null},
                {"id":"no-metrics","path":"/y","start":"","stop":"","port":8891,"slot":"aux-embedding","engine":"vllm","host":"192.168.20.63","containers":[],"process_match":null},
                {"id":"live-llm","path":"/z","start":"","stop":"","port":8888,"slot":"primary-llm","engine":"vllm","host":"192.168.20.63","containers":[],"process_match":null}
            ]"#,
        )
        .expect("registry fixture");

        let mut probes: HashMap<String, ProbeResult> = HashMap::new();
        probes.insert(
            "stopped-llm".into(),
            ProbeResult { ok: false, ..Default::default() },
        );
        probes.insert(
            "no-metrics".into(),
            ProbeResult { ok: true, concurrent: None, ..Default::default() },
        );
        probes.insert(
            "live-llm".into(),
            ProbeResult { ok: true, concurrent: Some(0), queued: Some(0), ..Default::default() },
        );

        assert_eq!(
            resolve_target("auto", &reg, &probes).map(|e| e.id),
            Some("live-llm".into())
        );
        assert_eq!(
            resolve_target("no-metrics", &reg, &probes).map(|e| e.id),
            Some("no-metrics".into()),
            "explicit ids are honoured even without telemetry"
        );
        assert!(resolve_target("nope", &reg, &probes).is_none());
    }

    #[test]
    fn auto_never_retargets_at_an_aux_service() {
        // Every primary-llm is down; the embed service is the only thing up.
        // The gate must stay on the GPU server and fail closed, not silently
        // watch bge-m3 and claim the box is idle.
        let reg: Vec<RegistryEntry> = serde_json::from_str(
            r#"[
                {"id":"llm-a","path":"/x","start":"","stop":"","port":8888,"slot":"primary-llm","engine":"vllm","host":"192.168.20.63","containers":[],"process_match":null},
                {"id":"embed","path":"/y","start":"","stop":"","port":8891,"slot":"aux-embedding","engine":"vllm","host":"192.168.20.63","containers":[],"process_match":null}
            ]"#,
        )
        .expect("registry fixture");

        let mut probes: HashMap<String, ProbeResult> = HashMap::new();
        probes.insert("llm-a".into(), ProbeResult { ok: false, ..Default::default() });
        probes.insert(
            "embed".into(),
            ProbeResult { ok: true, concurrent: None, ..Default::default() },
        );

        assert_eq!(
            resolve_target("auto", &reg, &probes).map(|e| e.id),
            Some("llm-a".into()),
            "a stopped GPU server must not fall back to an aux entry"
        );
    }

    #[test]
    fn exit_codes_are_stable() {
        assert_eq!(
            IdleOutcome::Open {
                entry: "e".into(),
                idle_for: Duration::from_secs(60),
                waited: Duration::from_secs(60)
            }
            .exit_code(),
            0
        );
        assert_eq!(
            IdleOutcome::TimedOut {
                entry: "e".into(),
                waited: Duration::from_secs(300),
                last: Sample::default()
            }
            .exit_code(),
            1
        );
        assert_eq!(
            IdleOutcome::UnknownEntry {
                entry: "nope".into(),
                known: vec![]
            }
            .exit_code(),
            2
        );
    }
}
