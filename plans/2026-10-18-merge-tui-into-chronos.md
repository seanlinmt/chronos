# Plan: merge `spark-deck` (TUI) into `spark-deck-http` → unified repo `chronos`

**Destination:** one repo `chronos` — a Cargo workspace with a single shared domain core and two frontends (TUI + HTTP), both frontends deployed and verified on `ai`, both git histories preserved, GitHub remote renamed.

## Current context (verified)

- `spark-deck` (TUI): `src/main.rs` 2239 lines, deps tokio/serde/reqwest/ratatui 0.29/crossterm 0.28/chrono. 3 commits, no remote. Deployed `ai:~/spark-deck`, run via `~/.local/bin/spark-deck`. Data: `~/spark-deck/data/registry.json`.
- `spark-deck-http` (axum): `src/main.rs` 1815 lines + `static/index.html` 851 lines (embedded via `include_str!`), deps axum 0.8/tokio/serde/reqwest. 1 commit (`966959f`), remote `git@github.com:seanlinmt/spark-deck-http.git`. Deployed `ai:~/spark-deck-http` as systemd user unit `spark-deck-http.service` (linger) on `:8090`. Data: `~/spark-deck-http/data/registry.json`.
- Both are **parallel implementations of the same domain** (storage gatekeeper, node metrics, GB10 per-PID GPU attribution, 3-engine telemetry, registry, detached lifecycle). The TUI version carries the verified `--dump-state` contract and the 401-status-check fix.
- Remote Rust is 1.95.0; local Mac 1.99 (`cargo check` OK, release builds on `ai` only).

## Target architecture

```
chronos/                          (renamed from spark-deck-http; GitHub: seanlinmt/chronos)
├── Cargo.toml                    [workspace] members = ["crates/*"]
├── crates/
│   ├── core/                     chronos-core — ALL domain logic, one source of truth
│   │   └── src/lib.rs            registry model+seed+load/save, storage gatekeeper,
│   │                             node metrics (meminfo/df/SSH blob), GB10 per-PID
│   │                             attribution, 3-engine probes (status-checked),
│   │                             state assembly (dump-state JSON shape), detached
│   │                             lifecycle, log tails, drop-caches
│   ├── tui/                      chronos-tui → bin chronos-tui (ratatui frontend,
│   │   └── src/main.rs           watch-channel consumer of core collectors + --dump-state)
│   └── http/                     chronos-http → bin chronos-http (axum frontend,
│       ├── src/main.rs           core-backed actions + /api/state adapter,
│       └── static/index.html     embedded web UI, unchanged behavior)
├── AGENTS.md                     merged ops contract for both frontends
└── README.md
```

Shared on `ai`: one deploy dir `~/chronos`, **one** registry `~/chronos/data/registry.json`, one log dir `~/chronos/logs/`. Service: `chronos-http.service` on `:8090` (same port). TUI: `~/.local/bin/chronos-tui` (+ transitional `spark-deck` symlink).

## Steps (each = one session-sized task)

### Step 0 — Safety net (local, 10 min)
1. Both trees clean: `git status --short` empty in both repos.
2. Backup branches: `git branch backup/pre-merge` in each.
3. Capture the live `/api/state` shape from `ai` (this is the web-UI compatibility oracle):
   `ssh ai 'curl -s http://127.0.0.1:8090/api/state' > /tmp/api-state-before.json` (with SSH retry loop).
4. Also capture `docker ps --format '{{.Names}}'` on `ai` as the untouched-containers baseline.

### Step 1 — Workspace skeleton in spark-deck-http (no behavior change)
1. `git mv src/main.rs crates/http/src/main.rs`; `git mv static crates/http/static`; fix `include_str!("../static/index.html")` → `include_str!("../static/index.html")` relative to the new location (`crates/http/static/...`).
2. Root `Cargo.toml` → `[workspace] resolver = "2"` + `members = ["crates/*"]`; new `crates/http/Cargo.toml`: name `chronos-http`, bin `chronos-http`, deps = existing four (axum 0.8, tokio full, serde, serde_json, reqwest 0.12 rustls no-default).
3. `cargo check` locally → green. Commit: `restructure: workspace skeleton, http crate at crates/http`.

### Step 2 — Merge TUI history (history preserved, still no unification)
1. `git remote add tui ../spark-deck && git fetch tui`.
2. `git merge tui/main --allow-unrelated-histories` — conflicts only on `Cargo.toml`/`README.md`/`AGENTS.md`/`.gitignore` (`src/main.rs` no longer collides; `LICENSE` identical). Resolve: root = workspace, README/AGENTS = keep HTTP versions for now, `.gitignore` = union.
3. Move merged TUI files into place: `src/main.rs` → `crates/tui/src/main.rs` + new `crates/tui/Cargo.toml` (name `chronos-tui`, bin `chronos-tui`, deps incl. ratatui 0.29/crossterm 0.28/chrono). Update `include`-free paths; TUI `base_dir()` currently hardcodes `/home/sean/spark-deck` — leave as-is until Step 3 retargets it.
4. `cargo check --workspace` → green. Commit: `merge spark-deck TUI (history) into crates/tui`.

### Step 3 — Extract `chronos-core` from the TUI implementation (the core refactor)
1. New `crates/core` (`chronos-core`, lib): move out of `crates/tui/src/main.rs` all domain sections (registry/seed/save, parsing helpers, storage/nodes/attribution/probe collectors, shared state + publishers, lifecycle actions). Frontend keeps only: TUI state, drawing, key handling, `main`.
2. `base_dir()` in core: replace hardcoded `/home/sean/spark-deck` with `/home/sean/chronos` (env override `CHRONOS_HOME` for tests/Mac runs, cwd fallback unchanged).
3. Core exposes: `Collector` handle (watch channel of snapshots), `assemble_snapshot()` (the `--dump-state` JSON), `start/stop/add/remove/drop_caches/logs` actions, registry load/seed/save.
4. `chronos-tui` becomes a thin consumer; `--dump-state` uses core's one-pass assemble.
5. Verify locally: `cargo check --workspace` green; `cargo run -p chronos-tui -- --dump-state` on Mac runs, exits 0, correct JSON shape (Mac nulls for `/proc` fields expected).
6. Commit: `extract chronos-core from TUI domain logic`.

### Step 4 — Rewire `chronos-http` onto core (delete its duplicated collectors)
1. Replace the HTTP crate's own collectors/state with core's; routes call core actions. Keep start/stop semantics identical (core already uses `process_group(0)` + `logs/<id>.log`).
2. **Web-UI compatibility gate:** core snapshot → `/api/state` adapter that reproduces the exact JSON key set captured in Step 0.3. Compare:
   `diff <(jq -S 'keys' /tmp/api-state-before.json) <(jq -S 'keys' <new>)` per object level (keys of `storage`, `nodes`, `entries[]`, etc.). Any key the web UI reads that core doesn't produce must be filled by the adapter.
3. Run the new `chronos-http` locally on an ephemeral port (`:18090`), feed it the captured state or run against live `ai` only from `ai`; on the Mac it starts but metrics are null — acceptable for shape check only (HTML served, `/api/state` key-complete).
4. `cargo check --workspace` green. Commit: `http frontend on chronos-core; /api/state shape preserved`.

### Step 5 — Rename to `chronos`
1. `mv /Users/sean/projects/spark-deck-http /Users/sean/projects/chronos`.
2. GitHub: `gh repo rename chronos -R seanlinmt/spark-deck-http` (fallback: rename in web UI, then `git remote set-url origin git@github.com:seanlinmt/chronos.git`).
3. Update README title + install snippets, AGENTS paths (see Step 8), LICENSE unchanged.
4. Commit + `git push origin main`.

### Step 6 — Deploy on `ai` (careful order; containers untouched)
1. Backups on `ai`: `cp -a ~/spark-deck-http/data ~/spark-deck-http-data.bak` and `cp -a ~/spark-deck/data ~/spark-deck-data.bak` (registry + logs preserved; keep old dirs in place until Step 7 passes).
2. Create `~/chronos/src crates data logs` (SSH retry loop; mkdir before scp), scp the whole workspace (Cargo.toml, crates/, README, AGENTS.md, LICENSE, .gitignore).
3. Detached build: `ssh ai 'export PATH="$HOME/.cargo/bin:$PATH"; cd ~/chronos && nohup cargo build --release > build.log 2>&1 &'` → poll until `Finished`.
4. Registry migration: `cp ~/spark-deck/data/registry.json ~/chronos/data/registry.json` (TUI's copy is the verified one; diff vs HTTP's — expect same 5 entries, reconcile manually if drifted).
5. TUI symlink: `ln -sf ~/chronos/target/release/chronos-tui ~/.local/bin/chronos-tui`; keep `~/.local/bin/spark-deck` → point it at the same target (transitional alias).
6. Service cutover (seconds of :8090 downtime): write `~/.config/systemd/user/chronos-http.service` (copy of the old unit with `WorkingDirectory=/home/sean/chronos`, `ExecStart=/home/sean/chronos/target/release/chronos-http`), `systemctl --user daemon-reload && systemctl --user stop spark-deck-http && systemctl --user disable spark-deck-http && systemctl --user enable --now chronos-http` (linger already on). Old unit file stays on disk for rollback.

### Step 7 — Verify (all gates green before cleanup)
1. `ssh ai 'cd ~/chronos && ./target/release/chronos-tui --dump-state | jq .'` — full TUI contract: storage online/all 5 checks/≈714 GiB avail; ai+ai2 `mem_total_bytes` ≈130 GB; tensorfold RUNNING ≈205–220 GB + non-null telemetry; **DeepSeek STOPPED "models http 401"** (correct steady state); bge entries RUNNING with `served_model`.
2. `ssh ai 'curl -s http://127.0.0.1:8090/api/state | jq .'` — key-set diff vs Step 0 capture empty; same live values; storage gate + entry states consistent with dump-state.
3. Web UI: fetch `http://192.168.20.63:8090/` — page loads, spot-check gauges/entry rows render (browser preview or `curl` + field grep).
4. Containers: `docker ps` names == Step 0 baseline, all 4 RUNNING, untouched.
5. `journalctl --user -u chronos-http -n 20` — no errors.
6. Rollback path (only if a gate fails): `systemctl --user stop chronos-http && systemctl --user enable --now spark-deck-http` (old dir was never modified); fix forward otherwise.

### Step 8 — Docs + closeout
1. Rewrite `chronos/AGENTS.md`: merged contract — workspace layout, crates, deploy flow for `~/chronos` (both binaries), registry/log paths, **both** verification gates (`--dump-state` + `:8090/api/state`), all hard-won quirks (SSH retry loop, GFS2/containers/detached-lifecycle constraints, status-code-before-parse rule, GB10 attribution method, single SSH blob for ai2, TUI-relaunch-to-pick-up-binary).
2. README: unified project intro, both frontends, workspace map.
3. Cleanup on `ai` (only after Step 7 green): `mv ~/spark-deck-http ~/spark-deck-http.retired`, `mv ~/spark-deck ~/spark-deck.retired` (keep `.bak` data copies; don't delete anything).
4. Local: archive `/Users/sean/projects/spark-deck` → `spark-deck.merged.bak` (its history lives in chronos now) — confirm with user before deleting.
5. Final `git push`.

## Risks & mitigations
- **`/api/state` shape drift breaks web UI silently** → Step 0 oracle + Step 4 key-set diff gate; adapter layer in http crate.
- **GB10 attribution behavioral drift** between the two old implementations → core is the TUI's (verified) version; Step 7.1/7.2 cross-check tensorfold ≈205 GiB via **both** frontends.
- **`:8090` cutover blip** → seconds, systemd stop/start in one command; old unit retained.
- **Rust 1.95 remote vs 1.99 local** → `cargo check` locally is a filter only; release build + `--dump-state` on `ai` is the real gate (as before).
- **Merge commit doubles as restructure** → avoided: restructure lands as normal commits (Steps 1–4); the history-merge commit itself is minimal (Step 2).

## Open questions / assumptions
- Binaries/services named `chronos-tui` / `chronos-http` / `chronos-http.service`; port stays `:8090`. (assumed yes)
- One shared registry at `~/chronos/data/registry.json` used by both frontends — a feature of unification, but means TUI and web see the same entry list. (assumed desired)
- `spark-deck` local dir: archive after push; deletion only on explicit user confirm. (assumed)
- `gh` CLI auth on this Mac is unverified — fallback path included in Step 5.
