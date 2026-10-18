# AGENTS.md — spark-deck

## What this repo is

**Staging repo only.** Source of truth is a running deployment elsewhere:

- Deployed to `ai:/home/sean/spark-deck/` (2-node DGX Spark GB10 cluster; `ai` = head via passwordless SSH, `ai2` = worker reached from `ai` via `ssh -o BatchMode=yes ai2`).
- Built and run **on `ai`**, never locally — the Mac has no guarantee of the Linux toolchain targets.
- Files mirror 1:1: `Cargo.toml`, `src/main.rs` (the entire app, single file), `data/registry.json` (seeded on first run), `logs/` (launcher output).

## Deployment flow (required order)

1. Edit files here.
2. `scp` changed files to `ai:/home/sean/spark-deck/` (same relative paths). **Create dest dirs first** (`ssh ai 'mkdir -p ~/spark-deck/src ~/spark-deck/data ~/spark-deck/logs'`) — scp fails with "No such file or directory" if `src/` doesn't exist yet.
3. Build detached — SSH drops mid-command otherwise:
   ```bash
   ssh ai 'export PATH="$HOME/.cargo/bin:$PATH"; cd ~/spark-deck && nohup cargo build --release > build.log 2>&1 & echo started'
   ```
4. Check `build.log` until `Finished`; on errors, fix here, re-scp, rebuild.
5. No service restart needed: the TUI is run interactively. The binary is exposed via `~/.local/bin/spark-deck → ~/spark-deck/target/release/spark-deck` (symlink already exists; rebuild replaces the target in place). There is **no** systemd unit and no HTTP server on :8090 — earlier web-dashboard iterations are gone.
6. Verify (see below).

## SSH gotcha

`ssh ai` intermittently fails with `ssh: connect to host ai.clearpixels.org port 22: Undefined error: 0` — transient, not a config problem. Wrap every SSH/SCP in a retry loop (`for i in 1 2 3; do ssh ... && break || sleep 6; done`). Do not conclude the host is down on first failure.

## Hard constraints (verified 2026-10)

- **Never** write anything to `/mnt/nvmeof` or `/home/sean/repos` — GFS2 over NVMe-oF; if it hangs/unmounts, spark-deck must still launch from local ext4. Build artifacts, registry, and logs live under `/home/sean/spark-deck/` (local disk). On the Mac, `base_dir()` falls back to cwd, so `--dump-state`/test runs write `./data/` and `./logs/` locally — that's fine.
- **Never** stop/restart containers `glm53-tf-r0`, `glm53-tf-r1`, `vllm_bge_m3`, `vllm_bge_reranker` — production inference servers; the dashboard only observes them.
- **Detached lifecycle is load-bearing**: every `start_cmd`/`stop_cmd` spawns via `sh -c` with `.process_group(0)`, stdin null, output appended to `logs/<id>.log`. Quitting the TUI (`q`) must never kill children. Don't change this to direct spawns.

## Rust quirks that already cost a build

- Deps are pinned by the deployed `Cargo.toml`: tokio full, serde/serde_json, reqwest 0.12 (rustls-tls, **no default features**), ratatui 0.29, crossterm 0.28, chrono 0.4. **Adding a crate means updating the remote `Cargo.toml` first** (remote Rust is 1.95.0; avoid crates needing newer).
- `std::sync::MutexGuard` held across `.await` breaks tokio's `Handler`/`Send` bounds — clone what you need, drop the guard, then await.
- GB10 unified-memory fact: `docker stats`/`VmRSS` report only ~4–7 GiB CPU RSS and miss ~98 GiB of CUDA unified memory. Device-level `nvidia-smi` says `Memory-Usage: Not Supported`, but per-process works: `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits`. True per-entry memory = Σ(GPU MiB + VmRSS) attributed via `/proc/<pid>/cgroup` `docker-<hex>.scope` ↔ `docker ps --no-trunc`, or `process_match` on cmdline.
- `ai2` metrics come from **one combined SSH blob per 3s tick** (6s overall timeout) — do not split into multiple SSH calls per tick.
- **Always check HTTP status before parsing a probe body** (`r.status().is_success()`). Multiple registry entries share `host:port` 8888, and TensorFold's auth-gated endpoints return **401 with a valid JSON error body** — parsing that body without a status check fakes `RUNNING` (this exact bug shipped once). Probes must treat non-2xx as failure with the status in `status_detail`.
- Rebuilding the release binary does **not** affect an already-running TUI session — the user must quit (`q`) and relaunch `spark-deck` to pick up a fix. Don't assume the running dashboard reflects the new binary.

## Verification contract

After any backend change, `--dump-state` is the test path. From `ai` (or `ssh ai`):

```bash
ssh ai 'cd ~/spark-deck && ./target/release/spark-deck --dump-state | jq .'
```

Expect (numbers drift with load; presence and rough magnitude are the contract):
- `storage.online == true` with all 5 checks true; `/mnt/nvmeof` avail ≈ 700+ GiB
- `ai`/`ai2` memory: `mem_total_bytes` ≈ 130 GB each, `mem_available_bytes` non-null
- `glm53-tensorfold-spark` RUNNING on :8888 with `mem_total_bytes` ≈ 205–220 GB and non-null telemetry (`concurrent_requests`, `total_tokens_generated`, `lifetime_decode_tok_per_sec`)
- `DeepSeek-v4.1-Flash-2x-DGX-Spark` **STOPPED with `status_detail` ≈ "models http 401 Unauthorized" is the correct steady state** — it shares port 8888 with the running TensorFold and is not launched. Do not "fix" this as a false negative; only its engine (or the auth) being up would make it RUNNING.
- `launch-embedding-ai.sh` (:8891) and `launch-reranker-ai2.sh` (:8892) RUNNING, ~1.7–2.0 GiB each, with non-null `served_model`

Local Mac `cargo check` (1.99) is fine for typechecking before the remote release build; runtime metrics are only real on `ai` (Mac runs will show nulls for `/proc`-derived fields — expected).
