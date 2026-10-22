//! Shared snapshot state, publisher loops, the `Collector` handle, and the
//! one-pass `--dump-state` assembly.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::sync::watch;

use crate::collectors::{
    build_client, collect_mem_attr, collect_nodes, collect_storage,
};
use crate::model::{
    AppState, EntrySnapshot, GraceKind, HTTP_INTERVAL_SECS, ProbeResult, RegistryEntry,
    STARTING_GRACE_SECS, SYS_INTERVAL_SECS, StorageState,
};
use crate::probes::{CollectorState, probe_all};
use crate::registry::load_or_seed_registry;

/// Mutable state shared between collectors, actions and the frontends.
pub struct Shared {
    pub tx: watch::Sender<Arc<AppState>>,
    pub registry: RwLock<Vec<RegistryEntry>>,
    pub storage: RwLock<StorageState>,
    pub nodes: RwLock<BTreeMap<String, Option<crate::model::NodeState>>>,
    /// id -> (mem_ai_bytes, mem_ai2_bytes)
    pub mem: RwLock<HashMap<String, (u64, u64)>>,
    pub probes: RwLock<HashMap<String, ProbeResult>>,
    pub states: RwLock<HashMap<String, String>>,
    pub grace: Mutex<HashMap<String, (GraceKind, Instant)>>,
    /// node -> previous /proc/stat (busy, total) sample for CPU load deltas.
    pub cpu_prev: std::sync::Mutex<BTreeMap<String, (u64, u64)>>,
}

impl Shared {
    pub fn publish(&self) {
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

pub fn assemble(
    registry: &[RegistryEntry],
    storage: &StorageState,
    nodes: &BTreeMap<String, Option<crate::model::NodeState>>,
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
// PUBLISHER LOOPS
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
        let nodes = collect_nodes(&shared.cpu_prev).await;
        let registry = shared.registry.read().unwrap().clone();
        let mem = collect_mem_attr(&registry).await;
        *shared.storage.write().unwrap() = storage;
        *shared.nodes.write().unwrap() = nodes;
        *shared.mem.write().unwrap() = mem;
        shared.publish();
    }
}

// =============================================================================
// COLLECTOR HANDLE
// =============================================================================

/// Handle to the background collector loops. Frontends subscribe via
/// `subscribe()`/`snapshot()` and call the lifecycle actions.
pub struct Collector {
    shared: Arc<Shared>,
    /// Kept alive so the channel never has zero receivers (otherwise every
    /// `send` would fail and new subscribers would only see the default).
    _rx: watch::Receiver<Arc<AppState>>,
}

impl Collector {
    /// Load (or seed) the registry, spawn the background loops, publish an
    /// initial snapshot.
    pub fn spawn() -> Collector {
        let registry = load_or_seed_registry();
        let (tx, rx) = watch::channel(Arc::new(AppState::default()));

        // Ensure the data/ and logs/ dirs exist before anything writes there.
        let _ = std::fs::create_dir_all(crate::paths::logs_dir());
        let _ = std::fs::create_dir_all(crate::paths::registry_path().parent().unwrap_or(std::path::Path::new(".")));

        let states: HashMap<String, String> = registry
            .iter()
            .map(|e| (e.id.clone(), "STOPPED".to_string()))
            .collect();
        let shared = Arc::new(Shared {
            tx,
            registry: RwLock::new(registry),
            storage: RwLock::new(StorageState::default()),
            nodes: RwLock::new(BTreeMap::new()),
            mem: RwLock::new(HashMap::new()),
            probes: RwLock::new(HashMap::new()),
            states: RwLock::new(states),
            grace: Mutex::new(HashMap::new()),
            cpu_prev: std::sync::Mutex::new(BTreeMap::new()),
        });
        // publish the initial snapshot so the first consumer frame has data
        shared.publish();

        let client = build_client();
        tokio::spawn(http_loop(shared.clone(), client));
        tokio::spawn(sys_loop(shared.clone()));

        Collector { shared, _rx: rx }
    }

    /// Watch channel of snapshots (TUI renders from this).
    pub fn subscribe(&self) -> watch::Receiver<Arc<AppState>> {
        self._rx.clone()
    }

    /// Latest published snapshot.
    pub fn snapshot(&self) -> Arc<AppState> {
        self._rx.borrow().clone()
    }

    /// Current registry (a clone).
    pub fn registry(&self) -> Vec<RegistryEntry> {
        self.shared.registry.read().unwrap().clone()
    }

    pub fn find(&self, id: &str) -> Option<RegistryEntry> {
        self.shared
            .registry
            .read()
            .unwrap()
            .iter()
            .find(|e| e.id == id)
            .cloned()
    }

    pub async fn start(&self, id: &str) -> Result<String, String> {
        crate::lifecycle::start_entry(id, &self.shared).await
    }

    pub async fn stop(&self, id: &str) -> Result<String, String> {
        crate::lifecycle::stop_entry(id, &self.shared).await
    }

    pub async fn add(&self, entry: RegistryEntry) -> Result<String, String> {
        let r = crate::lifecycle::add_entry(entry, &self.shared);
        self.shared.publish();
        r
    }

    pub async fn remove(&self, id: &str) -> Result<String, String> {
        let r = crate::lifecycle::remove_entry(id, &self.shared);
        self.shared.publish();
        r
    }

    pub async fn update(&self, old_id: &str, entry: RegistryEntry) -> Result<String, String> {
        let r = crate::lifecycle::update_entry(old_id, entry, &self.shared);
        self.shared.publish();
        r
    }

    pub async fn drop_caches(&self) -> crate::lifecycle::DropCachesResult {
        crate::lifecycle::drop_caches().await
    }

    /// TUI-style combined log view (file tail + docker logs per container).
    pub async fn entry_logs(&self, id: &str) -> Result<Vec<String>, String> {
        crate::lifecycle::entry_logs(id).await
    }
}

// =============================================================================
// ONE-PASS DUMP STATE (--dump-state contract)
// =============================================================================

/// Run one full collection pass and return the assembled state document.
pub async fn run_dump_state() -> Value {
    let client = build_client();
    let registry = load_or_seed_registry();
    let mut cs = CollectorState::default();

    let storage = collect_storage().await;
    // Seed a first /proc/stat sample so the one-pass dump can report ai's CPU
    // load (collect_storage above supplies the delta window). ai2 has no
    // previous sample in one-pass mode, so its load is None here.
    let cpu_prev: std::sync::Mutex<BTreeMap<String, (u64, u64)>> =
        std::sync::Mutex::new(BTreeMap::new());
    if let Some(sample) = crate::collectors::read_cpu_stat() {
        cpu_prev.lock().unwrap().insert("ai".into(), sample);
    }
    let nodes = collect_nodes(&cpu_prev).await;
    let mem = collect_mem_attr(&registry).await;
    let probes = probe_all(&registry, &mut cs.prev, &client).await;

    let mut states: HashMap<String, String> = HashMap::new();
    for e in &registry {
        let ok = probes.get(&e.id).map(|p| p.ok).unwrap_or(false);
        states.insert(e.id.clone(), if ok { "RUNNING" } else { "STOPPED" }.to_string());
    }

    let app = assemble(&registry, &storage, &nodes, &mem, &probes, &states);
    serde_json::to_value(&app).unwrap_or(Value::Null)
}
