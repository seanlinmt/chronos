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
5. TUI: no restart needed; `~/.local/bin/chronos` symlink points at `~/chronos/target/release/chronos`. HTTP: `ssh ai 'sudo systemctl restart chronos-http'`. Embeddings: `ssh ai 'sudo systemctl restart chronos-embed'` (or `./launch-embedding-ai.sh --force` from `/mnt/nvmeof/launchers`). Reranker: `ssh -o BatchMode=yes ai2 'sudo systemctl restart chronos-rerank'` (or `./launch-reranker-ai2.sh --force` from `ai`). The embed unit runs from `target/release`; the rerank unit on ai2 runs `/home/sean/chronos/bin/chronos-rerank` — after a build on `ai`, pipe the fresh binary to ai2 (`ssh ai 'cat target/release/chronos-rerank | ssh ai2 "cat > ~/chronos/bin/chronos-rerank.new"'`, hash it, then `mv`) before restarting. Resource-guard drop-ins (`chronos-{embed,rerank}.service.d/40-cpu-guard.conf`) need `sudo systemctl daemon-reload` before the restart picks up edits.
6. Verify — **both** gates below.

## SSH gotcha

`ssh ai` intermittently fails with `ssh: connect to host ai.clearpixels.org port 22: Undefined error: 0` — transient, not a config problem. Wrap every SSH/SCP in a retry loop (`for i in 1 2 3; do ssh ... && break || sleep 6; done`). Also note: remote commands ending in a command whose exit status is non-zero (e.g. bare `systemctl is-active`, `grep` with no match) will fail the whole SSH invocation — append `|| true` inside the remote command or the retry loop spins for nothing.

Two more that cost time (verified 2026-11):

- **`scp` to `ai` silently no-ops on this link** (`scp: Connection closed`, exit status hidden by the retry loop, remote file left at its old contents). Pipe instead and **verify the hash**: `ssh ai 'cat > ~/chronos/<path>' < <local file>` then compare `md5 -q` locally against `md5sum` remotely. Never trust a transferred file you have not hashed.
- **`pkill -f "cargo build"` kills its own remote shell**: the `bash -c` argv of the very command contains the pattern, so the SSH command dies before the later `echo`/`nohup` runs (the bracket trick `[c]argo` does not help when the same command line also contains the literal pattern elsewhere). Don't kill builds by pattern — start a new one with a fresh log file and let cargo's target-dir lock serialize.

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
- **veloGB10 counters are completion-time, not incremental** (verified 2026-11 on v0.7.8): `/health` telemetry reads `mode:"none"` with `mtp.emitted = dflash2.emitted = 0` and `yield_tok_per_step: null` when speculation is off, and `velogb10_generation_tokens_total` / `_prompt_tokens_total` count **finished requests only** — measured flat for 8 s with 4 lanes generating. So a live rate derived from either reads `null` during steady-state generation. `live_tok_per_sec` comes from the delta of `velogb10_scheduler_steps_total` × `requests_running` (1 token per lane per step when speculation is off), with `step_ms_p50` and the finished-token delta as lower-priority fallbacks — see `velogb10_live_rate` in `crates/core/src/probes.rs`. `lifetime_decode_tok_per_sec` is `decode_tokens_per_second_sum / _count`, which the server documents as the **per-request** rate, while `live_tok_per_sec` is the **aggregate** over lanes (same convention as the vLLM probe): with N lanes the two differ by ~N, and that is correct, not a bug. `wait-idle` fail-closes on a null gauge, so `requests_running == 0` must report `Some(0.0)` and never `null`, or the gate can never open.
- **CPU-only aux services must pin their pools *and* be cgroup-fenced** (2026-11 incident): `#[tokio::main]` defaults to `worker_threads = num_cores` (20 on GB10) plus a blocking pool that grows toward 512 threads, idle ones kept alive ~10 s past their last task — chronos-embed grew 28 → 48 OS threads over 13 days and an embedding burst during GPU inference drove CPU to 94.8 °C. Both ONNX bins now build their runtime explicitly (`TOKIO_WORKER_THREADS` / `TOKIO_MAX_BLOCKING_THREADS`, read via `env_num` — **tokio reads no env vars itself**, a unit `Environment=TOKIO_*=` line only works because the binary parses it) and gate ONNX calls with `Semaphore::acquire_owned()` (permits = 1); the model mutex already serializes inference, so past the blocking cap tasks *queue* instead of spawning threads. Independent of the code, `/etc/systemd/system/chronos-{embed,rerank}.service.d/40-cpu-guard.conf` fences the units: `CHRONOS_*_THREADS=4` + `OMP_NUM_THREADS=4` + `ORT_INTRA/INTER_OP_NUM_THREADS` + `TOKIO_*`, `CPUQuota=350%`, `CPUWeight=20`, `MemoryHigh=2G`/`MemoryMax=3G`/`MemorySwapMax=0`. GB10 topology (verified `lscpu -e` on both nodes): efficiency cores are **0-4 / 10-14 @ 2808 MHz**, perf cores **5-9 / 15-19 @ 3900 MHz** — `lscpu -e`'s CORE column is interleaved between clusters, don't infer ranges from it. Aux units pin `CPUAffinity=0-3`. When adding any CPU-only aux service: same pattern — pin pools in code, fence with a drop-in.
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

ssh ai 'cd ~/chronos && ./target/release/chronos wait-idle --for 30s --max-wait 0'
# --max-wait 0 (or `forever`) = no deadline: wait however long the box stays busy,
# for work queued behind a job that may run for hours. Ctrl-C to abort.
```

- Targets `--entry auto` by default: the `primary-llm` slot entry that reports telemetry (slot-first, so a model swap never retargets the gate at an aux embedding entry; a stopped GPU server reads as closed, not idle).
- Requires **`concurrent_requests == 0` AND `queued_requests == 0` AND `live_tok_per_sec ≈ 0`** for every sample across the window. `/api/state` gauges are 2 s point samples; the tok/s term is a delta over `generation_tokens_total`, so a flat window also means no tokens were generated between polls. Caveat: only the TensorFold probe synthesises a zero rate for a flat counter — the plain vLLM probe reports `null` unless the counter moved, and `null` is tolerated when both gauges read zero, so for vLLM entries only the two gauges are load-bearing.
- `--max-wait` defaults to **300 s**; on a busy box the gate exits 1 after that. Use `0`/`forever` for hour-scale queues, and a `--heartbeat` (default 60 s, `0` disables) keeps the wait visible — it fires at most once per `--sample` tick, so a heartbeat shorter than the sample period is a no-op.
- **Fail-closed**: a null gauge (probe failed, entry stopped/renamed, `/metrics` disabled, CPU-only entry) resets the window. `--allow-unknown` opts out — only for CPU/ONNX entries, never for the GPU server.
- `--json` prints one machine-readable result object; progress lines go to stderr.
- `scripts/idle-gate.sh` is the same policy over `/api/state` for hosts without the binary (needs curl+jq).

## Verification contract

**Gate 1 — TUI (`--dump-state`):**
```bash
ssh ai 'cd ~/chronos && ./target/release/chronos --dump-state | jq .'
```
- `ai`/`ai2` `mem_total_bytes` ≈ 130 GB each, `mem_available_bytes` non-null
- The **active `primary-llm` entry** (the one that owns `:8888`) is RUNNING, with `mem_total_bytes` non-null in the ~200 GB range (GPU unified memory included) and non-null telemetry (`concurrent_requests`, `total_tokens_generated`, `lifetime_decode_tok_per_sec`).
- Every other entry on `:8888` is **STOPPED**, with a `status_detail` that explains why (HTTP status / model mismatch). Only one server can hold the port, so this is the correct steady state — do not "fix" it, and never stop a running inference server to make a gate pass.
- The aux services are RUNNING: `:8891` on `ai` (`chronos-embed`, CPU/ONNX int8) with `mem_total_bytes` ≈ **1.1 GiB**, and `:8892` on `ai2` (`chronos-rerank`, CPU/ONNX fp32) with ≈ **1.7 GiB**; both report a non-null `served_model`. These are plain process RSS — CPU-only services never appear in the nvidia-smi compute-apps list, so no CUDA unified memory is added. Since the 40-cpu-guard drop-ins (swap off, `MemoryHigh` charging), mmapped ONNX pages stay resident and count in RSS — expect roughly the model weight size, still under the unit's 3 GiB cap. Their footprint follows the loaded weight set, so re-measure rather than assuming the previous numbers. Field names differ by gate: `--dump-state` uses `state`, `/api/state` uses `status` (plus `name` as legacy alias of `id`).

Entry ids rotate with model swaps, so check by slot and port — never by name.

**Gate 2 — HTTP:**
```bash
ssh ai 'curl -s http://127.0.0.1:8090/api/state | jq ".entries[] | {id, status, served_model, mem_total_bytes}"'
curl -s -o /dev/null -w "%{http_code}" http://192.168.20.63:8090/   # 200, ~70 KB page
```
- Same statuses as Gate 1; entry objects carry all 19 keys incl. `telemetry.*` (`name` is a legacy alias of `id`; see `entry_dto` in `crates/http/src/main.rs`); top-level `throttle_baseline_ts` + `nodes.<n>.throttle` come from the throttle baseline; `journalctl -u chronos-http` shows no errors.

Local Mac runs show null `/proc`-derived fields — expected; runtime metrics are only real on `ai`. Rebuilding does **not** update an already-running TUI session — the user must quit (`q`) and relaunch.
