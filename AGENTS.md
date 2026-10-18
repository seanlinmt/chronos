# AGENTS.md — spark-deck-http

## What this repo is

**Staging repo only.** The source of truth for a running service that lives elsewhere:

- Deployed to `ai:/home/sean/spark-deck-http/` (2-node DGX Spark cluster; `ai` = head via passwordless SSH, `ai2` = worker reached from `ai`).
- Built and run **on `ai`**, never locally — this Mac has no guarantee of cargo/Rust targets.
- Files mirror 1:1: `Cargo.toml`, `src/main.rs` (entire server, single file), `static/index.html` (embedded via `include_str!`), `LICENSE`, `README.md`.

## Deployment flow (required order)

1. Edit files here.
2. `scp` changed files to `ai:/home/sean/spark-deck-http/` (same relative paths; `src/main.rs` → `src/main.rs`, `static/index.html` → `static/index.html`).
3. Build detached — SSH drops mid-command otherwise:
   ```bash
   ssh ai 'export PATH="$HOME/.cargo/bin:$PATH"; cd ~/spark-deck-http && nohup cargo build --release > build.log 2>&1 & echo started'
   ```
4. Check `build.log` until `Finished`; on errors, fix here, re-scp, rebuild.
5. Restart if the binary changed: `ssh ai 'systemctl --user restart spark-deck-http'`.
6. Verify: `curl -s http://127.0.0.1:8090/api/state | jq .` from `ai` (see README for expected shape).

## SSH gotcha

`ssh ai` intermittently fails with `ssh: connect to host ai.<cluster-domain> port 22: Undefined error: 0` — transient, not a config problem. Wrap every SSH/SCP in a retry loop (`for i in 1 2 3; do ssh ... && break || sleep 6; done`). Do not conclude the host is down on first failure.

## Hard constraints (verified 2026-10)

- **Never** write anything to `/mnt/nvmeof` or `/home/sean/repos` — GFS2 over NVMe-oF; if it hangs/unmounts the dashboard must stay up (it lives on local ext4 `/dev/nvme0n1p2`). Registry (`data/registry.json`) and logs are relative paths resolved from the service `WorkingDirectory=/home/sean/spark-deck-http`.
- **Never** stop/restart containers `glm53-tf-r0`, `glm53-tf-r1`, `vllm_bge_m3`, `vllm_bge_reranker` — they are production inference servers; the dashboard only observes them.
- Service: user systemd `spark-deck-http.service` (linger enabled). Manage with `systemctl --user {status,restart,stop}`; logs via `journalctl --user -u spark-deck-http -f`.

## Rust quirks that already cost a build

- Deps are pinned by the deployed `Cargo.toml`: axum 0.8, tokio full, serde/serde_json, reqwest 0.12 (rustls-tls, **no default features**). Adding a crate means updating the remote `Cargo.toml` first.
- axum 0.8 path params are `/{id}`, **not** `/:id`.
- `std::sync::MutexGuard` held across `.await` breaks the `Handler` trait (future not `Send`) — clone what you need, drop the guard, then await. This produced the E0277 `Handler<_, _>` errors before.
- `include_str!("../static/index.html")` fails the build if the static file is missing.

## Verification contract

After any backend change, via `ssh ai`:
- `storage.online == true` with all 5 checks true
- `glm53-tensorfold-spark` RUNNING with `mem_total_bytes` ≈ 205–220 GB and non-null telemetry (`concurrent_requests`, `total_tokens_generated`)
- `launch-embedding-ai-bge-m3` (ai, ~1.7 GB) and `launch-reranker-ai2-bge-reranker` (ai2, ~1.9 GB) RUNNING
- `curl -s http://127.0.0.1:8090/ | head -c 50` serves the embedded HTML

Numbers drift with load; presence and rough magnitude are the contract, not exact values. GPU memory ~100 GiB per tensorfold rank is correct on GB10 (`docker stats`/`VmRSS` miss it — that's why per-PID `nvidia-smi --query-compute-apps` attribution exists).
