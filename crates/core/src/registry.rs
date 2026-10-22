//! Registry: seed entries, load-or-seed from `<base>/data/registry.json`, save.

use crate::model::{AI2_LAN, AI_LAN, RegistryEntry};
use crate::paths::registry_path;

pub fn seed_registry() -> Vec<RegistryEntry> {
    vec![
        RegistryEntry {
            id: "glm53-tensorfold-spark".into(),
            path: "/home/sean/repos/glm53-tensorfold-spark".into(),
            start: "./launch.sh start".into(),
            stop: "./launch.sh stop".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "tensorfold".into(),
            containers: vec!["glm53-tf-r0".into(), "glm53-tf-r1".into()],
            process_match: None,
        },
        RegistryEntry {
            id: "DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            path: "/home/sean/repos/DeepSeek-v4.1-Flash-2x-DGX-Spark".into(),
            start: "./start-server.sh".into(),
            stop: "./stop-server.sh".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "vllm".into(),
            containers: vec![],
            process_match: None,
        },
        RegistryEntry {
            id: "veloGB10 (Qwen3.8-Flash-Next)".into(),
            path: "/home/sean/repos/veloGB10".into(),
            start: "./launch_qwen38_flash_next.sh 2".into(),
            stop: "pkill -f gb10_inference || true; ssh -o BatchMode=yes ai2 'pkill -f gb10_inference || true'".into(),
            host: AI_LAN.into(),
            port: 8888,
            slot: "primary-llm".into(),
            engine: "velogb10".into(),
            containers: vec![],
            process_match: Some("gb10_inference".into()),
        },
        RegistryEntry {
            id: "launch-embedding-ai.sh (BGE-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start: "./launch-embedding-ai.sh --force".into(),
            stop: "docker rm -f vllm_bge_m3".into(),
            host: AI_LAN.into(),
            port: 8891,
            slot: "aux-embedding".into(),
            engine: "vllm".into(),
            containers: vec!["vllm_bge_m3".into()],
            process_match: None,
        },
        RegistryEntry {
            id: "launch-reranker-ai2.sh (BGE-Reranker-v2-M3)".into(),
            path: "/mnt/nvmeof/launchers".into(),
            start: "./launch-reranker-ai2.sh --force".into(),
            stop: "ssh -o BatchMode=yes ai2 'docker rm -f vllm_bge_reranker'".into(),
            host: AI2_LAN.into(),
            port: 8892,
            slot: "aux-reranker".into(),
            engine: "vllm".into(),
            containers: vec!["vllm_bge_reranker".into()],
            process_match: None,
        },
    ]
}

/// Load the registry from `<base>/data/registry.json`; seed + persist the
/// default 5 entries when the file is missing or unparseable.
pub fn load_or_seed_registry() -> Vec<RegistryEntry> {
    let p = registry_path();
    if let Ok(s) = std::fs::read_to_string(&p) {
        if let Ok(v) = serde_json::from_str::<Vec<RegistryEntry>>(&s) {
            return v;
        }
    }
    let seeded = seed_registry();
    let _ = save_registry(&seeded);
    seeded
}

pub fn save_registry(entries: &[RegistryEntry]) -> Result<(), String> {
    let p = registry_path();
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    }
    let s = serde_json::to_string_pretty(entries).map_err(|e| e.to_string())?;
    std::fs::write(&p, s).map_err(|e| format!("write {}: {e}", p.display()))
}
