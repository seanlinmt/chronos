# Spark Deck HTTP — GX10 Cluster Dashboard

A lightweight, production-ready Rust (`axum` + `tokio`) control and telemetry dashboard for a
2-node NVIDIA DGX Spark (Grace Blackwell GB10) cluster: `ai` (head, 192.168.20.63) and
`ai2` (worker, 192.168.20.229).

Serves a dark single-page UI at `http://ai:8090/` and a JSON API under `/api/*`.

## What it does

- **Shared storage gatekeeper** — checks `/mnt/nvmeof` (GFS2 over NVMe-oF RAID0 `md0`) on
  **both** nodes every 3s with a strict 2s timeout (hung GFS2/DLM never blocks the server).
  When offline, Start / Add-Repo operations return `409` and are disabled in the UI.
- **Per-node metrics (3s)** — unified memory (total / used / available / page cache / swap)
  and local root-disk (`/`) usage for `ai` and `ai2`, plus shared-storage capacity. `ai2`
  is polled via a single combined SSH one-liner per cycle.
- **GB10 unified-memory accounting** — `docker stats` and `VmRSS` miss ~98 GiB of CUDA
  unified memory on GB10, so per-process GPU memory is read from
  `nvidia-smi --query-compute-apps` and combined with host RSS. PIDs are mapped to
  endpoints via container ID (`/proc/<pid>/cgroup`), working directory, and process match.
- **Normalized telemetry (2s)** across three engines:
  | Engine | Source | Metrics |
  |---|---|---|
  | TensorFold | `GET /health` | in-flight, lifetime + Δ-decode tok/s, prompt/completion/cached token totals |
  | vLLM | `GET /metrics` + `/v1/models` | running/waiting, generation tokens, Δ tok/s; `http_requests_total` fallback when `--disable-log-stats` |
  | veloGB10 | `GET /health` | mtp+dflash2 emitted tokens, Δ tok/s with `yield/step_ms` fallback |
- **Repo & launcher registry** with start/stop lifecycle, per-slot conflict guards
  (one active `primary-llm`), log capture, and auto-detect scaffolding.

## Install (on `ai`)

Toolchain: Rust 1.95.0 at `$HOME/.cargo/bin`. Everything lives on **local ext4**
(`/dev/nvme0n1p2`) — never on `/mnt/nvmeof` or `/home/sean/repos`.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p ~/spark-deck-http/src ~/spark-deck-http/static
# copy Cargo.toml, src/main.rs, static/index.html, LICENSE into ~/spark-deck-http/
cd ~/spark-deck-http
cargo build --release
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/spark-deck-http.service <<EOF
[Unit]
Description=GX10 DGX Spark Cluster Dashboard (axum)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/sean/spark-deck-http
ExecStart=/home/sean/spark-deck-http/target/release/spark-deck-http
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now spark-deck-http
sudo loginctl enable-linger $USER   # survive logout/reboot
```

## Running / managing

```bash
systemctl --user status spark-deck-http
systemctl --user restart spark-deck-http
systemctl --user stop spark-deck-http
journalctl --user -u spark-deck-http -f
```

The server binds `0.0.0.0:8090`. On boot it seeds
`data/registry.json` (relative to the working directory) with the 5 cluster entries on
first run and appends start/stop output to `logs/<id>.log`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | embedded single-page UI |
| GET | `/api/state` | full snapshot: storage health, node memory/disk, entries with live status, memory attribution, telemetry |
| POST | `/api/entries` | register a repo/launcher (`409` on duplicate id) |
| POST | `/api/entries/inspect` | `{"path": ...}` → auto-detected start/stop commands, ports, containers |
| DELETE | `/api/entries/{id}` | remove registry entry (never touches files) |
| POST | `/api/entries/{id}/start` | `409` if storage offline or another `primary-llm` is running |
| POST | `/api/entries/{id}/stop` | run `stop_cmd`, append to log |
| GET | `/api/entries/{id}/logs` | last 200 log lines, or `docker logs --tail 100` for running containers |
| POST | `/api/cluster/drop-caches` | `sudo -n sync && sudo -n sysctl -w vm.drop_caches=3` on both nodes |

## Seeded entries

| ID | Slot | Port | Engine |
|---|---|---|---|
| `glm53-tensorfold-spark` | primary-llm | 8888 | tensorfold |
| `deepseek-v4.1-flash-2x-dgx-spark` | primary-llm | 8888 | vllm |
| `velogb10-qwen38-flash-next` | primary-llm | 8888 | velogb10 |
| `launch-embedding-ai-bge-m3` | aux-embedding | 8891 | vllm |
| `launch-reranker-ai2-bge-reranker` | aux-reranker | 8892 | vllm |

## Layout

```
~/spark-deck-http/
├── Cargo.toml
├── src/main.rs          # entire server (axum 0.8, tokio, reqwest rustls)
├── static/index.html    # embedded UI (no CDNs, vanilla JS)
├── data/registry.json   # persistent registry (atomic writes)
├── logs/<id>.log        # start/stop output
├── LICENSE              # MIT
└── README.md
```

## Notes

- All subprocess/HTTP calls are timeout-guarded (`tokio::time::timeout`); a hung SSH or
  GFS2 stat degrades data instead of stalling the API.
- `ai2` metric failures keep last-known node data; storage checks report the precise
  failing node+check in `storage.reason`.
- Do not stop `glm53-tf-r0`, `glm53-tf-r1`, `vllm_bge_m3`, or `vllm_bge_reranker` while
  verifying — the dashboard only observes them by default.

## License

MIT — see [LICENSE](LICENSE).
