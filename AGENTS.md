# AGENTS.md — chronos

## What this repo is

**Staging repo only.** Source of truth is the deployment on `ai`:

- 2-node DGX Spark GB10 cluster: `ai` = head (passwordless SSH), `ai2` = worker (reached from `ai` via `ssh -o BatchMode=yes ai2`). GitHub: `seanlinmt/chronos` (renamed from spark-deck-http; remote URL updated).
- Built and run **on `ai`**, never locally — the Mac has no guarantee of the Linux toolchain targets. Local Mac `cargo check` (1.99) is a typecheck filter only.
- Cargo workspace, four crates:
  - `crates/core` — ALL domain logic (registry, GB10 per-PID GPU attribution, 3-engine probes, snapshot assembly, detached lifecycle). One source of truth; domain fixes land here once.
  - `crates/tui` — `chronos` bin (NOT `chronos-tui`; that name is retired): ratatui dashboard + `--dump-state` JSON mode. UI/keys only; domain via core.
  - `crates/http` — `chronos-http` bin: axum on `:8090` + `static/index.html` (embedded, `include_str!`). Runs as **system-scope** systemd unit on `ai`: `/etc/systemd/system/chronos-http.service` (`sudo systemctl restart chronos-http`). The adapter maps core snapshot → the exact `/api/state` key set the web UI reads. **Do not change the shape** without diffing against the JS in `static/index.html`.
  - `crates/embed` — two bins sharing `src/lib.rs`: `chronos-embed` (BGE-M3 embeddings, fastembed / ONNX int8, CPU-only, dense 1024-d) on `:8891`, and `chronos-rerank` (bge-reranker-v2-m3, fastembed / ONNX **fp32**, CPU-only, `/v1/rerank`) on `:8892`. Each runs as a **system-scope** systemd unit (`chronos-embed` on `ai`, `chronos-rerank` on `ai2`). The registry start paths `/mnt/nvmeof/launchers/launch-embedding-ai.sh` and `launch-reranker-ai2.sh` drive those units. They replace the retired `vllm_bge_m3` / `vllm_bge_reranker` containers. Note: fastembed returns raw logits for rerank, so `chronos-rerank` applies a sigmoid to match vLLM's 0–1 `relevance_score`.

## Deployment flow (required order)

1. Edit files here.
2. scp changed files to `ai:~/chronos/` mirroring the crate layout (`crates/core/src/*.rs`, `crates/tui/src/main.rs`, `crates/http/src/main.rs`, `crates/http/static/index.html`, `crates/embed/src/main.rs`, root manifests). Create dirs first if missing.
3. Build detached — SSH drops mid-command otherwise:
   ```bash
   ssh ai 'export PATH="$HOME/.cargo/bin:$PATH"; cd ~/chronos && nohup cargo build --release > build.log 2>&1 & echo started'
   ```
4. Check `build.log` until `Finished`; on errors, fix here, re-scp, rebuild.
5. TUI: no restart needed; `~/.local/bin/chronos` symlink points at `~/chronos/target/release/chronos`. HTTP: `ssh ai 'sudo systemctl restart chronos-http'`. Embeddings: `ssh ai 'sudo systemctl restart chronos-embed'` (or `./launch-embedding-ai.sh --force` from `/mnt/nvmeof/launchers`). Reranker: `ssh -o BatchMode=yes ai2 'sudo systemctl restart chronos-rerank'` (or `./launch-reranker-ai2.sh --force` from `ai`).
6. Verify — **both** gates below.

## SSH gotcha

`ssh ai` intermittently fails with `ssh: connect to host ai.clearpixels.org port 22: Undefined error: 0` — transient, not a config problem. Wrap every SSH/SCP in a retry loop (`for i in 1 2 3; do ssh ... && break || sleep 6; done`). Also note: remote commands ending in a command whose exit status is non-zero (e.g. bare `systemctl is-active`, `grep` with no match) will fail the whole SSH invocation — append `|| true` inside the remote command or the retry loop spins for nothing.

## Hard constraints (verified 2026-10)

- Everything lives under `/home/sean/chronos` (local ext4). **Never** write to `/mnt/nvmeof` or `/home/sean/repos` (GFS2; must not be a launch dependency) — with the deliberate exceptions of regenerable **caches**: `/mnt/nvmeof/vllm-cache/embedding` (Triton/JIT cache) and `/mnt/nvmeof/fastembed-cache` (ONNX model). Both are caches only (a missing mount costs a rebuild/download, not a failure), and the root fs is at ~100% so they cannot live there. Registry: `~/chronos/data/registry.json`; logs: `~/chronos/logs/<id>.log`. Old installs `~/spark-deck` and `~/spark-deck-http` were retired, not deleted.
- One **shared registry** — TUI and web see the same entry list. Editing it via either frontend affects both.
- **Never** stop/restart containers `glm53-tf-r0`, `glm53-tf-r1` — production inference servers; chronos only observes them. (`vllm_bge_m3` and `vllm_bge_reranker` are retired: embeddings and reranking now run natively as the `chronos-embed` and `chronos-rerank` systemd units.)
- **Detached lifecycle is load-bearing**: start/stop spawn via `sh -c` with `.process_group(0)`, stdin null, output appended to `logs/<id>.log` (in `crates/core/src/lifecycle.rs`). Quitting the TUI or restarting the service must never kill children.
- `Collector::spawn()` must keep one watch receiver alive internally — dropping it makes every publish fail and subscribers freeze on the default snapshot (this bug was caught during the merge).

## Rust quirks that already cost a build

- Deps pinned in per-crate `Cargo.toml`s + workspace `Cargo.toml`: tokio full, serde/serde_json, reqwest 0.12 (rustls-tls, **no default features**), axum 0.8 (http), ratatui 0.29/crossterm 0.28 (tui), chrono. **Adding a crate means updating the remote manifests first** (remote Rust is 1.95.0; avoid crates needing newer).
- `std::sync::MutexGuard` held across `.await` breaks tokio/axum `Send` bounds — clone what you need, drop the guard, then await.
- **Always check HTTP status before parsing a probe body** (`r.status().is_success()` — enforced in all 3 engine probes now). Entries share `host:port` 8888; TensorFold's auth-gated endpoints return **401 with a valid JSON error body** — parsing without a status check fakes `RUNNING` (this exact bug shipped once). Non-2xx → fail with the status in `status_detail`.
- axum 0.8 path params are `/{id}`, **not** `/:id`.
- GB10 unified-memory fact: `docker stats`/`VmRSS` show only ~4–7 GiB CPU RSS and miss ~98 GiB of CUDA unified memory. Device-level `nvidia-smi` says `Memory-Usage: Not Supported`, but per-process works: `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits`. True per-entry memory = Σ(GPU MiB + VmRSS), attributed via `/proc/<pid>/cgroup` `docker-<hex>.scope` ↔ `docker ps --no-trunc`, or `process_match` on cmdline. CPU-only services (no CUDA allocation) never appear in the compute-apps list, so they are attributed by a second pass that discovers PIDs with `pgrep -f <process_match>`; **only entries that set `process_match` opt into that pass** — never widen it to fallback-matched entries or unrelated processes get swept in.
- `ai2` metrics come from **one combined SSH blob per 3s tick** (6s overall timeout) — do not split into multiple SSH calls per tick.
- Env overrides: `CHRONOS_HOME` (base dir, for Mac/test runs), `CHRONOS_HTTP_PORT` (bind port, default 8090).

## Agent / cron gating (`chronos wait-idle`)

Before running anything that competes with production inference, gate on idleness:

```bash
ssh ai 'cd ~/chronos && ./target/release/chronos wait-idle --for 60s --max-wait 300s'
# exit 0 = GPU entry has been idle for the whole window → safe to proceed
# exit 1 = gate closed (never held the window) → do not proceed
# exit 2 = unknown entry / usage error
```

- Targets `--entry auto` by default: the `primary-llm` slot entry that reports telemetry (slot-first, so a model swap never retargets the gate at `bge-m3`; a stopped GPU server reads as closed, not idle).
- Requires **`concurrent_requests == 0` AND `queued_requests == 0` AND `live_tok_per_sec ≈ 0`** for every sample across the window. `/api/state` gauges are 2 s point samples; the tok/s term is a delta over `generation_tokens_total`, so a flat window also means no tokens were generated between polls.
- **Fail-closed**: a null gauge (probe failed, entry stopped/renamed, `/metrics` disabled, CPU-only entry) resets the window. `--allow-unknown` opts out — only for CPU/ONNX entries, never for the GPU server.
- `--json` prints one machine-readable result object; progress lines go to stderr.
- `scripts/idle-gate.sh` is the same policy over `/api/state` for hosts without the binary (needs curl+jq).

## Verification contract

**Gate 1 — TUI (`--dump-state`):**
```bash
ssh ai 'cd ~/chronos && ./target/release/chronos --dump-state | jq .'
```
- `ai`/`ai2` `mem_total_bytes` ≈ 130 GB each, `mem_available_bytes` non-null
- `glm53-tensorfold-spark` RUNNING, `mem_total_bytes` ≈ 205–220 GB, non-null telemetry (`concurrent_requests`, `total_tokens_generated`, `lifetime_decode_tok_per_sec`)
- `DeepSeek-v4.1-Flash-2x-DGX-Spark` **STOPPED with `status_detail` ≈ "models http 401 Unauthorized" is the correct steady state** — it shares port 8888 with the running TensorFold. Do not "fix" this.
- bge-m3 (:8891) RUNNING via `chronos-embed` (CPU/ONNX int8), `mem_total_bytes` ≈ **0.3 GiB** (325 MB — current embedding weight set; earlier builds reported 0.9–1.1 GiB); bge-reranker-v2-m3 (:8892) RUNNING via `chronos-rerank` on `ai2` (CPU/ONNX fp32), ≈ **1.2 GiB** (1.26 GB, was documented as 1.6–1.8 GiB); both with non-null `served_model`. These are plain process RSS — CPU-only services never appear in the nvidia-smi compute-apps list, so no CUDA unified memory is added.

**Gate 2 — HTTP:**
```bash
ssh ai 'curl -s http://127.0.0.1:8090/api/state | jq ".entries[] | {id, status, served_model, mem_total_bytes}"'
curl -s -o /dev/null -w "%{http_code}" http://192.168.20.63:8090/   # 200, ~32KB page
```
- Same statuses as Gate 1; entry objects carry all 18 keys incl. `telemetry.*` (see the adapter in `crates/http/src/main.rs`); `journalctl -u chronos-http` shows no errors.

Local Mac runs show null `/proc`-derived fields — expected; runtime metrics are only real on `ai`. Rebuilding does **not** update an already-running TUI session — the user must quit (`q`) and relaunch.
