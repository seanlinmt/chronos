# chronos

Lightweight dashboard suite for a 2-node NVIDIA DGX Spark (Grace Blackwell GB10) cluster (`ai` head + `ai2` worker). One shared domain core, two frontends:

```
chronos/
├── crates/core   — domain logic: registry, storage gatekeeper, GB10 memory
│                   attribution, 3-engine telemetry, detached lifecycle
├── crates/tui    — chronos-tui: ratatui terminal dashboard + --dump-state
└── crates/http   — chronos-http: axum web dashboard (:8090) + embedded UI
```

![Chronos Dashboard](assets/dashboard.png)

## What it does

- **Storage gatekeeper** — verifies `/mnt/nvmeof` (GFS2 over NVMe-oF RAID0 `md0`, repos/huggingface paths) on both nodes with 2s per-check timeouts; a hung DLM lock never freezes either frontend. Blocks start/add while offline.
- **Unified memory + disk gauges** — per-node `MemTotal/Available/Cache/Swap` and root-disk use with explicit Available.
- **GB10-correct per-server memory** — `docker stats`/`VmRSS` miss ~98 GiB of CUDA unified memory, so memory = `nvidia-smi --query-compute-apps` (GPU MiB) + `/proc/<pid>/status` RSS, attributed via docker cgroup scopes and `process_match`.
- **3-engine telemetry, normalized** — TensorFold (`/health`), vLLM (`/v1/models` + Prometheus `/metrics`, status-checked), veloGB10 (`/health` MTP/dflash2): concurrent requests, live tok/s, lifetime decode tok/s, total tokens.
- **Detached lifecycle** — start/stop run in their own session (`process_group(0)`, logs → `logs/<id>.log`); quitting the TUI or restarting the service never touches running inference servers. One active `primary-llm` slot guard.
- **Shared registry** — both frontends see the same entries (`data/registry.json`, seeded with the cluster's 5 launchers on first run).

## The two frontends

**TUI** — 4-zone ratatui dashboard: top status bar with GFS2 badge, dual-node memory/disk gauges, active-endpoint telemetry, registry table. Keys: `j/k` select · `s` start · `x` stop (`y/n`) · `l` logs · `a` add (auto-detects `launch.sh`/`start-*.sh`/`launch_*.sh` + `.env*`) · `d` remove · `c` drop caches · `r` refresh · `q` quit (leaves servers running).

```bash
chronos
chronos --dump-state | jq .   # headless JSON snapshot, exit 0
```

**Web** — dark single-page UI served at `http://ai:8090/` with the same data over `/api/*` (list/add/remove/start/stop/logs/drop-caches). Runs as a systemd user service (`chronos-http.service`, linger enabled).

## Build

Rust 1.95+, no system deps (reqwest uses rustls):

```bash
cargo build --release          # workspace: core, tui, http
```

Deployment target is the head node at `~/chronos`; see [AGENTS.md](AGENTS.md) for the scp + detached-build flow, systemd unit, and the two-gate verification contract.

## License

[MIT](LICENSE)
