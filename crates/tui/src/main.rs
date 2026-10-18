// =============================================================================
// chronos-tui — ratatui frontend for the 2-node NVIDIA DGX Spark (GB10)
// cluster (head: `ai`, worker: `ai2`).
//
// Thin consumer of chronos-core: background collectors, registry, engine
// probes and lifecycle actions all live in the core crate and publish
// snapshots through a tokio::sync::watch channel. This file keeps only:
//   1. TUI state (selection, modal, confirm, log drawer)
//   2. drawing
//   3. key handling
//   4. action dispatch -> core
//   5. main (+ --dump-state via core::run_dump_state)
// =============================================================================

use std::path::Path;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Table, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;

use chronos_core::{
    compact, gib, Collector, RegistryEntry, AI2_LAN, AI_LAN,
};

// =============================================================================
// 1. TUI STATE
// =============================================================================

#[derive(Default)]
struct AddModal {
    path: String,
    id: String,
    port: String,
    slot: String,
    engine: String,
    focus: usize,
    note: Option<String>,
    detected_start: Option<String>,
    detected_stop: Option<String>,
}

impl AddModal {
    fn new() -> Self {
        AddModal {
            path: String::new(),
            id: String::new(),
            port: "8888".into(),
            slot: "primary-llm".into(),
            engine: "vllm".into(),
            focus: 0,
            note: None,
            detected_start: None,
            detected_stop: None,
        }
    }
    fn fields(&mut self) -> [&mut String; 5] {
        [
            &mut self.path,
            &mut self.id,
            &mut self.port,
            &mut self.slot,
            &mut self.engine,
        ]
    }
    fn build(&self) -> RegistryEntry {
        RegistryEntry {
            id: if self.id.trim().is_empty() {
                "unnamed-entry".into()
            } else {
                self.id.trim().to_string()
            },
            path: self.path.trim().to_string(),
            start: self.detected_start.clone().unwrap_or_default(),
            stop: self.detected_stop.clone().unwrap_or_default(),
            host: if self.path.starts_with("/mnt/nvmeof") {
                AI2_LAN
            } else {
                AI_LAN
            }
            .to_string(),
            port: self.port.trim().parse().unwrap_or(8888),
            slot: self.slot.trim().to_string(),
            engine: self.engine.trim().to_string(),
            containers: vec![],
            process_match: None,
        }
    }
}

enum Confirm {
    Stop(String),
    Remove(String),
}

#[derive(Default)]
struct UiState {
    selected: usize,
    message: Option<(String, Instant)>,
    should_quit: bool,
    confirm: Option<Confirm>,
    add_modal: Option<AddModal>,
    log_view: Option<(String, Vec<String>)>,
}

enum Action {
    Start(String),
    Stop(String),
    Add(RegistryEntry),
    Remove(String),
    DropCaches,
    ShowLog(String),
}

enum UiMsg {
    Message(String),
    Log { id: String, lines: Vec<String> },
}

// --- add-modal script/env auto-detection ------------------------------------

fn detect_launch(path: &str) -> (Option<String>, Option<String>, Option<String>, Option<String>) {
    let dir = Path::new(path);
    let names: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    let mut port = None;
    let mut model = None;
    for envname in names.iter().filter(|n| n.starts_with(".env")) {
        if let Ok(txt) = std::fs::read_to_string(dir.join(envname)) {
            for line in txt.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let k = k.trim().to_ascii_uppercase();
                    let v = v.trim();
                    if (k.contains("PORT") || k == "PORT") && port.is_none() {
                        port = Some(v.to_string());
                    }
                    if (k.contains("MODEL") || k.contains("SERVED")) && model.is_none() {
                        model = Some(v.to_string());
                    }
                }
            }
        }
    }
    let has = |n: &str| names.iter().any(|x| x == n);
    let first_prefix = |p: &str| {
        names
            .iter()
            .filter(|x| x.starts_with(p))
            .min()
            .cloned()
    };
    let (start, stop) = if has("launch.sh") {
        (
            Some("./launch.sh start".to_string()),
            Some("./launch.sh stop".to_string()),
        )
    } else if has("start-server.sh") {
        (
            Some("./start-server.sh".to_string()),
            Some(if has("stop-server.sh") {
                "./stop-server.sh".to_string()
            } else {
                String::new()
            }),
        )
    } else if let Some(s) = first_prefix("start-") {
        (Some(format!("./{s}")), Some(String::new()))
    } else if let Some(s) = first_prefix("launch_") {
        (Some(format!("./{s}")), Some(String::new()))
    } else {
        (None, None)
    };
    (start, stop, port, model)
}

// =============================================================================
// 2. DRAWING
// =============================================================================

const KEYBAR: &str = "s start · x stop · l logs · a add · d del · c drop-caches · r refresh · q/Esc quit";

fn draw(f: &mut Frame, app: &chronos_core::AppState, ui: &UiState) {
    let area = f.area();
    let chunks = Layout::vertical([
        Constraint::Length(3), // top bar
        Constraint::Length(9), // dual-node panel
        Constraint::Min(6),    // active endpoints
        Constraint::Length(9), // registry table
        Constraint::Length(1), // footer
    ])
    .split(area);

    draw_top_bar(f, app, chunks[0]);
    draw_nodes(f, app, chunks[1]);
    draw_endpoints(f, app, chunks[2]);
    draw_table(f, app, ui, chunks[3]);
    draw_footer(f, ui, app, chunks[4]);

    if let Some((_, lines)) = &ui.log_view {
        draw_log_drawer(f, lines, bottom_half(chunks[2]));
    }
    if let Some(c) = &ui.confirm {
        draw_confirm(f, c, area);
    }
    if let Some(m) = &ui.add_modal {
        draw_add_modal(f, m, area);
    }
}

fn draw_top_bar(f: &mut Frame, app: &chronos_core::AppState, area: ratatui::layout::Rect) {
    let st = &app.storage;
    let badge = if st.online {
        let used = st.used_bytes.unwrap_or(0);
        let total = st.total_bytes.unwrap_or(0);
        let avail = st.avail_bytes.unwrap_or(0);
        Span::styled(
            format!(
                " [ONLINE: {}G / {}T ({}G avail)]",
                used / 1_073_741_824,
                total / 1_099_511_627_776,
                avail / 1_073_741_824
            ),
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            format!(" [OFFLINE: {}]", st.reason.as_deref().unwrap_or("checks failed")),
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )
    };
    let lines = vec![
        Line::from(vec![
            Span::styled(
                "SPARK-DECK // GB10 CLUSTER CONTROL",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            badge,
        ]),
        Line::from(Span::styled(
            if st.online {
                "GFS2 gate passed on ai + ai2 — start/remove allowed"
            } else {
                "GFS2 gate FAILED — `s` (start) and `a` (add) are blocked"
            },
            Style::default().fg(if st.online { Color::Green } else { Color::Red }),
        )),
        Line::from(Span::styled(
            format!("last update: {}", app.timestamp),
            Style::default().fg(Color::DarkGray),
        )),
    ];
    f.render_widget(Paragraph::new(lines), area);
}

fn mem_gauge_label(node: &Option<chronos_core::NodeState>) -> (String, f64, Style) {
    match node {
        Some(n) => {
            let total = n.mem_total_bytes.unwrap_or(0);
            let used = n.mem_used_bytes.unwrap_or(0);
            let avail = n.mem_available_bytes.unwrap_or(0);
            let cache = n.mem_cache_bytes.unwrap_or(0);
            let swap = n.swap_used_bytes.unwrap_or(0);
            let ratio = if total > 0 {
                (used as f64 / total as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let style = if ratio > 0.9 {
                Style::default().fg(Color::Red)
            } else if ratio > 0.8 {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default().fg(Color::Green)
            };
            (
                format!(
                    "Used {} / {} | Avail {} | Cache {} | Swap {}",
                    gib(used),
                    gib(total),
                    gib(avail),
                    gib(cache),
                    gib(swap)
                ),
                ratio,
                style,
            )
        }
        None => ("unreachable (ssh)".into(), 0.0, Style::default().fg(Color::DarkGray)),
    }
}

fn draw_node_half(f: &mut Frame, name: &str, node: &Option<chronos_core::NodeState>, area: ratatui::layout::Rect) {
    let block = Block::default().borders(Borders::ALL).title(name.to_string());
    let inner = block.inner(area);
    f.render_widget(block, area);
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)]).split(inner);
    let (label, ratio, style) = mem_gauge_label(node);
    f.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label(label.clone())
            .gauge_style(style),
        rows[0],
    );
    let disk = node.as_ref().and_then(|n| n.disk_root.as_ref());
    let (dlabel, dratio, dstyle) = match disk {
        Some(d) => {
            let r = if d.total_bytes > 0 {
                (d.used_bytes as f64 / d.total_bytes as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let color = if d.use_pct > 90.0 {
                Color::Red
            } else if d.use_pct > 80.0 {
                Color::Yellow
            } else {
                Color::Green
            };
            (
                format!(
                    "/ {} used | {} avail | {}%",
                    gib(d.used_bytes),
                    gib(d.avail_bytes),
                    d.use_pct as u64
                ),
                r,
                Style::default().fg(color),
            )
        }
        None => ("/ unavailable".into(), 0.0, Style::default().fg(Color::DarkGray)),
    };
    f.render_widget(
        Gauge::default()
            .ratio(dratio)
            .label(dlabel.clone())
            .gauge_style(dstyle),
        rows[2],
    );
    f.render_widget(Paragraph::new(label).style(Style::default()), rows[1]);
    f.render_widget(Paragraph::new(dlabel), rows[3]);
}

fn draw_nodes(f: &mut Frame, app: &chronos_core::AppState, area: ratatui::layout::Rect) {
    let halves = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);
    let ai = app.nodes.get("ai").cloned().flatten();
    let ai2 = app.nodes.get("ai2").cloned().flatten();
    draw_node_half(f, "ai (head · 192.168.20.63)", &ai, halves[0]);
    draw_node_half(f, "ai2 (worker · 192.168.20.229)", &ai2, halves[1]);
}

fn draw_endpoints(f: &mut Frame, app: &chronos_core::AppState, area: ratatui::layout::Rect) {
    let mut lines: Vec<Line> = Vec::new();
    let active: Vec<&chronos_core::EntrySnapshot> = app
        .entries
        .iter()
        .filter(|e| e.state != "STOPPED")
        .collect();
    if active.is_empty() {
        lines.push(Line::from(Span::styled(
            "no active endpoints",
            Style::default().fg(Color::DarkGray),
        )));
    }
    for e in active {
        let (dot, color) = match e.state.as_str() {
            "RUNNING" => ("●", Color::Green),
            "STARTING" | "STOPPING" => ("◐", Color::Yellow),
            _ => ("○", Color::DarkGray),
        };
        lines.push(Line::from(vec![
            Span::styled(format!("{dot} "), Style::default().fg(color)),
            Span::styled(&e.id, Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!(" ({}) — {}:{}", e.engine, e.host, e.port)),
            Span::styled(format!(" [{}]", e.state), Style::default().fg(color)),
        ]));
        let mut parts = vec![format!("mem {} ", gib(e.mem_total_bytes))];
        parts.push(format!(
            "(ai: {} | ai2: {}) · ",
            gib(e.mem_ai_bytes),
            gib(e.mem_ai2_bytes)
        ));
        if let Some(c) = e.concurrent_requests {
            let q = e.queued_requests.unwrap_or(0);
            parts.push(format!("conc {c} (+{q} q) · "));
        }
        if let Some(l) = e.live_tok_per_sec {
            parts.push(format!("{l:.1} tok/s (live) · "));
        }
        if let Some(a) = e.lifetime_decode_tok_per_sec {
            parts.push(format!("{a:.1} tok/s (decode avg) · "));
        }
        if let Some(t) = e.total_tokens_generated {
            let mut tail = format!("tot {}", compact(t));
            if let Some(p) = e.prompt_tokens_total {
                tail.push_str(&format!(" (prompt {})", compact(p)));
            }
            if let Some(c) = e.cached_tokens_total {
                tail.push_str(&format!(" (cache {})", compact(c)));
            }
            parts.push(tail);
        }
        if let Some(m) = &e.served_model {
            parts.push(format!(" · model {m}"));
        }
        lines.push(Line::from(Span::raw(parts.concat())));
        if let Some(d) = &e.status_detail {
            lines.push(Line::from(Span::styled(
                format!("  ⚠ {d}"),
                Style::default().fg(Color::Yellow),
            )));
        }
    }
    f.render_widget(
        Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("ACTIVE ENDPOINTS"))
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_table(f: &mut Frame, app: &chronos_core::AppState, ui: &UiState, area: ratatui::layout::Rect) {
    let header = Row::new(["Status", "Name", "Slot", "Port", "Mem", "Path"]).style(
        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    );
    let rows = app.entries.iter().enumerate().map(|(i, e)| {
        let (dot, color) = match e.state.as_str() {
            "RUNNING" => ("● RUNNING", Color::Green),
            "STARTING" | "STOPPING" => ("◐ STARTING", Color::Yellow),
            _ => ("○ STOPPED", Color::DarkGray),
        };
        let selected = i == ui.selected;
        let mut style = Style::default();
        if selected {
            style = style.add_modifier(Modifier::REVERSED);
        }
        Row::new(vec![
            Cell::from(dot).style(Style::default().fg(color).patch(style)),
            Cell::from(e.id.clone()).style(style),
            Cell::from(e.slot.clone()).style(style),
            Cell::from(e.port.to_string()).style(style),
            Cell::from(gib(e.mem_total_bytes)).style(style),
            Cell::from(e.path.clone()).style(style),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(12),
            Constraint::Min(24),
            Constraint::Length(14),
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Min(20),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("REGISTERED ENTRIES"));
    let mut ts = ratatui::widgets::TableState::default();
    ts.select(Some(ui.selected));
    f.render_stateful_widget(table, area, &mut ts);
}

fn draw_footer(f: &mut Frame, ui: &UiState, app: &chronos_core::AppState, area: ratatui::layout::Rect) {
    let mut spans = vec![Span::styled(KEYBAR, Style::default().fg(Color::DarkGray))];
    if let Some((m, at)) = &ui.message {
        if at.elapsed() < Duration::from_secs(8) {
            spans.push(Span::styled(
                format!("  ⟶ {m}"),
                Style::default()
                    .fg(if m.contains("fail") || m.contains("blocked") { Color::Red } else { Color::Green })
                    .add_modifier(Modifier::BOLD),
            ));
        }
    }
    if !app.storage.online {
        spans.push(Span::styled("  [GFS2 OFFLINE]", Style::default().fg(Color::Red)));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn bottom_half(area: ratatui::layout::Rect) -> ratatui::layout::Rect {
    let h = area.height / 2;
    ratatui::layout::Rect {
        x: area.x,
        y: area.y + area.height - h,
        width: area.width,
        height: h,
    }
}

fn draw_log_drawer(f: &mut Frame, lines: &[String], area: ratatui::layout::Rect) {
    f.render_widget(Clear, area);
    let text: Vec<Line> = lines.iter().map(|l| Line::from(l.clone())).collect();
    f.render_widget(
        Paragraph::new(text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("LOG TAIL (press any key to close)")
                    .style(Style::default().bg(Color::Black)),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_confirm(f: &mut Frame, c: &Confirm, area: ratatui::layout::Rect) {
    let (text, title) = match c {
        Confirm::Stop(id) => (format!("stop `{id}`? (y/n)"), "CONFIRM STOP"),
        Confirm::Remove(id) => (
            format!("remove `{id}` from registry? (y/n) — files are NOT deleted"),
            "CONFIRM REMOVE",
        ),
    };
    let w = 60.min(area.width);
    let h = 5;
    let r = centered(area, w, h);
    f.render_widget(Clear, r);
    f.render_widget(
        Paragraph::new(vec![Line::from(text)])
            .block(Block::default().borders(Borders::ALL).title(title))
            .alignment(Alignment::Center),
        r,
    );
}

fn draw_add_modal(f: &mut Frame, m: &AddModal, area: ratatui::layout::Rect) {
    let r = centered(area, 70, 12);
    f.render_widget(Clear, r);
    let labels = ["Path", "ID", "Port", "Slot", "Engine"];
    let values = [&m.path, &m.id, &m.port, &m.slot, &m.engine];
    let mut lines: Vec<Line> = Vec::new();
    for (i, (l, v)) in labels.iter().zip(values.iter()).enumerate() {
        let focused = i == m.focus;
        lines.push(Line::from(vec![
            Span::styled(format!("{l:>6}: "), Style::default().fg(Color::Cyan)),
            if focused {
                Span::styled(format!("{v}█"), Style::default().bg(Color::DarkGray))
            } else {
                Span::raw((*v).clone())
            },
        ]));
    }
    if let Some(n) = &m.note {
        lines.push(Line::from(Span::styled(
            format!("note: {n}"),
            Style::default().fg(Color::Yellow),
        )));
    }
    lines.push(Line::from(Span::styled(
        "Tab next field (auto-detects scripts/.env from Path) · Enter save · Esc cancel",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title("ADD REPOSITORY ENTRY"),
        ),
        r,
    );
}

fn centered(area: ratatui::layout::Rect, w: u16, h: u16) -> ratatui::layout::Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    ratatui::layout::Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

// =============================================================================
// 3. KEY HANDLING
// =============================================================================

fn handle_key(
    key: crossterm::event::KeyEvent,
    ui: &mut UiState,
    app: &chronos_core::AppState,
    action_tx: &mpsc::UnboundedSender<Action>,
) {
    if key.kind != KeyEventKind::Press {
        return;
    }
    // Log drawer open: any key closes it.
    if ui.log_view.is_some() {
        ui.log_view = None;
        return;
    }
    // Add modal consumes keys.
    if let Some(m) = ui.add_modal.as_mut() {
        match key.code {
            KeyCode::Esc => ui.add_modal = None,
            KeyCode::Tab => {
                if m.focus == 0 {
                    // leaving the path field: try auto-detection
                    let (start, stop, port, model) = detect_launch(&m.path);
                    if let Some(s) = start {
                        m.note = Some(format!("detected start `{s}`{}; port={:?} model={:?}", stop.map(|_| "").unwrap_or_default(), port, model));
                        m.detected_start = Some(s);
                    } else {
                        m.note = Some("no launch.sh/start-*.sh/launch_*.sh found (paths are remote dirs on `ai`)".into());
                        m.detected_start = None;
                    }
                    if let Some(p) = port {
                        m.port = p;
                    }
                    let _ = model;
                }
                m.focus = (m.focus + 1) % 5;
            }
            KeyCode::Enter => {
                if m.focus == 0 {
                    let (start, _stop, port, model) = detect_launch(&m.path);
                    if let Some(s) = start {
                        m.detected_start = Some(s);
                        m.note = Some("detected launch script".into());
                    } else {
                        m.note = Some("no launch script detected; entry will have no start command".into());
                    }
                    if let Some(p) = port {
                        m.port = p;
                    }
                    let _ = model;
                } else {
                    let entry = m.build();
                    ui.add_modal = None;
                    let _ = action_tx.send(Action::Add(entry));
                }
            }
            KeyCode::Backspace => {
                let idx = m.focus;
                m.fields()[idx].pop();
            }
            KeyCode::Char(c) => {
                let idx = m.focus;
                m.fields()[idx].push(c);
            }
            _ => {}
        }
        return;
    }
    // Confirm dialog consumes keys.
    if let Some(c) = ui.confirm.take() {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => match c {
                Confirm::Stop(id) => {
                    let _ = action_tx.send(Action::Stop(id));
                }
                Confirm::Remove(id) => {
                    let _ = action_tx.send(Action::Remove(id));
                }
            },
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {}
            _ => {
                // restore if unhandled key
                ui.confirm = Some(c);
            }
        }
        return;
    }

    let n = app.entries.len();
    match key.code {
        KeyCode::Char('q') => ui.should_quit = true,
        KeyCode::Down | KeyCode::Char('j') => {
            if n > 0 {
                ui.selected = (ui.selected + 1).min(n - 1);
            }
        }
        KeyCode::Up | KeyCode::Char('k') => ui.selected = ui.selected.saturating_sub(1),
        KeyCode::Char('r') => {
            // force refresh: handled by setting a flag via message; collector ticks are short anyway
            ui.message = Some(("refresh requested (collectors run every 2-3s)".into(), Instant::now()));
        }
        KeyCode::Char('s') => {
            if let Some(e) = app.entries.get(ui.selected) {
                if !app.storage.online {
                    ui.message = Some((
                        "start blocked: GFS2 storage offline".into(),
                        Instant::now(),
                    ));
                } else if e.slot == "primary-llm"
                    && app.entries.iter().any(|o| {
                        o.id != e.id && o.slot == "primary-llm" && o.state == "RUNNING"
                    })
                {
                    ui.message = Some((
                        "start blocked: another primary-llm entry is already RUNNING".into(),
                        Instant::now(),
                    ));
                } else {
                    let _ = action_tx.send(Action::Start(e.id.clone()));
                }
            }
        }
        KeyCode::Char('x') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.confirm = Some(Confirm::Stop(e.id.clone()));
            }
        }
        KeyCode::Char('d') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.confirm = Some(Confirm::Remove(e.id.clone()));
            }
        }
        KeyCode::Char('a') => {
            if !app.storage.online {
                ui.message = Some(("add blocked: GFS2 storage offline".into(), Instant::now()));
            } else {
                ui.add_modal = Some(AddModal::new());
            }
        }
        KeyCode::Char('c') => {
            let _ = action_tx.send(Action::DropCaches);
            ui.message = Some(("drop caches requested…".into(), Instant::now()));
        }
        KeyCode::Char('l') => {
            if let Some(e) = app.entries.get(ui.selected) {
                ui.log_view = Some((e.id.clone(), vec!["loading…".into()]));
                let _ = action_tx.send(Action::ShowLog(e.id.clone()));
            }
        }
        KeyCode::Esc => ui.should_quit = true,
        _ => {}
    }
}

// =============================================================================
// 4. ACTION DISPATCH -> CORE
// =============================================================================

async fn run_action(action: Action, collector: &Collector, ui_tx: &mpsc::UnboundedSender<UiMsg>) {
    match action {
        Action::Start(id) => {
            let msg = match collector.start(&id).await {
                Ok(m) => m,
                Err(e) => e,
            };
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::Stop(id) => {
            let msg = match collector.stop(&id).await {
                Ok(m) => m,
                Err(e) => e,
            };
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::Add(entry) => {
            let msg = collector.add(entry).await.unwrap_or_else(|e| e);
            // Next publish tick picks up the new entry.
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::Remove(id) => {
            let msg = collector.remove(&id).await.unwrap_or_else(|e| e);
            let _ = ui_tx.send(UiMsg::Message(msg));
        }
        Action::DropCaches => {
            let r = collector.drop_caches().await;
            let _ = ui_tx.send(UiMsg::Message(format!(
                "drop_caches: ai {} | ai2 {}",
                r.ai, r.ai2
            )));
        }
        Action::ShowLog(id) => {
            let lines = match collector.entry_logs(&id).await {
                Ok(lines) => lines,
                Err(err) => vec![format!("(cannot read log: {err})")],
            };
            let _ = ui_tx.send(UiMsg::Log { id, lines });
        }
    }
}

// =============================================================================
// 5. TERMINAL LIFECYCLE + MAIN
// =============================================================================

struct TermGuard;
impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stdout(), LeaveAlternateScreen);
    }
}

async fn run_tui() {
    let collector = Arc::new(Collector::spawn());
    let rx = collector.subscribe();
    let (action_tx, action_rx) = mpsc::unbounded_channel::<Action>();
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiMsg>();

    let actions_collector = collector.clone();
    tokio::spawn(async move {
        run_action_loop(action_rx, actions_collector, ui_tx.clone()).await;
    });

    let _guard = TermGuard;
    enable_raw_mode().expect("enable_raw_mode");
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).expect("enter alternate screen");
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).expect("terminal");

    let mut ui = UiState::default();
    loop {
        let app = rx.borrow().clone();
        // apply selection bound
        if !app.entries.is_empty() {
            ui.selected = ui.selected.min(app.entries.len() - 1);
        } else {
            ui.selected = 0;
        }
        terminal
            .draw(|f| draw(f, &app, &ui))
            .expect("terminal draw");

        // drain UI messages
        while let Ok(m) = ui_rx.try_recv() {
            match m {
                UiMsg::Message(text) => ui.message = Some((text, Instant::now())),
                UiMsg::Log { id, lines } => ui.log_view = Some((id, lines)),
            }
        }

        if event::poll(Duration::from_millis(250)).expect("event poll") {
            if let Ok(Event::Key(key)) = event::read() {
                if ui.should_quit {
                    break;
                }
                handle_key(key, &mut ui, &app, &action_tx);
            }
        }
        if ui.should_quit {
            break;
        }
    }
    // TermGuard restores raw mode + screen.
}

use std::sync::Arc;

async fn run_action_loop(
    mut rx: mpsc::UnboundedReceiver<Action>,
    collector: Arc<Collector>,
    ui_tx: mpsc::UnboundedSender<UiMsg>,
) {
    while let Some(a) = rx.recv().await {
        run_action(a, &collector, &ui_tx).await;
    }
}

#[tokio::main]
async fn main() {
    if std::env::args().any(|a| a == "--dump-state") {
        let state = chronos_core::run_dump_state().await;
        println!(
            "{}",
            serde_json::to_string_pretty(&state).unwrap_or_else(|_| "{}".to_string())
        );
        return;
    }
    run_tui().await;
}
