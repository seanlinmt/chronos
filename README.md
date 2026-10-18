# spark-deck

Lightweight terminal dashboard for a 2-node NVIDIA DGX Spark (Grace Blackwell GB10) cluster (`ai` head + `ai2` worker). Single-file Rust: `ratatui` + `crossterm` + `tokio`.

```
SPARK-DECK // GB10 CLUSTER CONTROL  [ONLINE: 585G / 1T (713G avail)]
┌─ ai (head) ─────────────────────┐ ┌─ ai2 (worker) ─────────────────┐
│ Used 113.5 / 121.6 GiB  Avail 8.1│ │ Used 112.0 / 121.6 GiB  Avail …│
├─ ACTIVE ENDPOINTS ──────────────┴─┴────────────────────────────────┤
│ ● glm53-tensorfold-spark — :8888 [RUNNING] mem 205.3 GiB 53.9 tok/s│
├─ REGISTERED ENTRIES ───────────────────────────────────────────────┤
│ ● RUNNING  glm53-tensorfold-spark  primary-llm  8888  205.3 GiB    │
└─────────────────────────────────────────────────────────────────────┘
 s start · x stop · l logs · a add · d del · c drop-caches · r refresh
```

## Features

- **Storage gatekeeper** — verifies `/mnt/nvmeof` (GFS2 over NVMe-oF, `md0` raid0, repos/huggingface paths) on both nodes with 2s per-check timeouts; a hung DLM lock never freezes the UI. Blocks `Start`/`Add` while offline.
- **Unified memory + disk gauges** — per-node `MemTotal/Available/Cache/Swap` and root-disk use (yellow >80%, red >90%) with explicit Available.
- **GB10-correct per-server memory** — `docker stats`/`VmRSS` miss ~98 GiB of CUDA unified memory, so memory is computed from `nvidia-smi --query-compute-apps` (GPU MiB) + `/proc/<pid>/status` (RSS), attributed via docker cgroup scopes and `process_match`.
- **3-engine telemetry, normalized** — TensorFold (`/health`), vLLM (`/v1/models` + Prometheus `/metrics`), and veloGB10 (`/health` MTP/dflash2) all report concurrent requests, live tok/s, lifetime decode tok/s, and total tokens generated.
- **Detached lifecycle** — `start`/`stop` run in their own session (`process_group(0)`, logs → `logs/<id>.log`); quitting the TUI never touches running inference servers.
- **One SSH round-trip per tick** for the remote node (single combined command batch).
- **`--dump-state`** — one full collection pass, pretty JSON to stdout, exit 0. CI/hook-friendly verification.

## Usage

```bash
spark-deck                    # TUI
spark-deck --dump-state | jq  # headless state snapshot
```

### Keys

| Key | Action |
|---|---|
| `↑/↓`, `j/k` | select entry |
| `s` | start (blocked while storage offline, or if another `primary-llm` is running) |
| `x` | stop (with `y/n` confirm) |
| `l` | live log drawer (entry log + `docker logs --tail 50` per container) |
| `a` | add repo/launcher (auto-detects `launch.sh`, `start-*.sh`, `launch_*.sh`, `.env*` ports/models) |
| `d` | remove entry from registry (`y/n`; never deletes files) |
| `c` | drop page caches on both nodes |
| `r` | force refresh |
| `q` / `Esc` | quit (leaves all servers running) |

## Registry

`data/registry.json` is seeded on first run with the cluster's 5 entries (tensorfold, DeepSeek-v4.1, veloGB10, BGE-M3 embedder, BGE-reranker). Each entry: `id`, `path`, `start`/`stop` cmds, `host:port`, `slot` (`primary-llm`/`aux-*`), `engine`, optional `containers` and `process_match`. State (RUNNING/STARTING/STOPPED) is derived live from HTTP probes — it is not stored.

## Build

Rust 1.95+, no system deps (reqwest uses rustls):

```bash
cargo build --release
```

Deployment target is a head node at `~/spark-deck` (see [AGENTS.md](AGENTS.md) for the scp + detached-build flow and the `--dump-state` verification contract).

## License

[MIT](LICENSE)
