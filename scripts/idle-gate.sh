#!/usr/bin/env bash
# idle-gate.sh — exit 0 only if a chronos GPU entry has had NO active requests,
# NO queued requests, and NO live decoding, continuously for WINDOW seconds.
#
#   ./scripts/idle-gate.sh [entry-id|auto] [window-secs]
#   defaults: auto 60
#
#   auto = pick the RUNNING entry that actually reports telemetry (the entry
#          bound to :8888), so model swaps don't break the gate.
#
# Prefer `chronos wait-idle` (same policy in core, no curl/jq needed); this
# script is the fallback for hosts that cannot run the binary.
#
# Env: CHRONOS_URL   default http://192.168.20.63:8090
#      SAMPLE_SECS   poll period, default 2 (chronos re-probes every 2s — faster is pointless)
#      MAX_WAIT_SECS total budget, default 300
#
# Second gate input: chronos probes every HTTP_INTERVAL_SECS=2s and each sample
# is a POINT GAUGE, so a sub-2s burst between polls is invisible. "idle" is
# strong evidence, not proof. Verify with prompts_completed / token counters if
# you need certainty that a job actually finished.
#
# Fail-closed: null gauges (probe failed, entry gone, metrics disabled) reset
# the window. Pass --allow-unknown as the 2nd/3rd arg to count them as idle.
# CPU/ONNX entries (bge-m3, bge-reranker-v2-m3) never report telemetry, so a
# gate on them can only ever pass with --allow-unknown.

set -uo pipefail

ENTRY="${1:-auto}"
WINDOW="${2:-60}"
URL="${CHRONOS_URL:-http://192.168.20.63:8090}"
SAMPLE="${SAMPLE_SECS:-2}"
ALLOW_UNKNOWN=0
case "${2:-}" in --allow-unknown) ALLOW_UNKNOWN=1; WINDOW=60;; esac
case "${3:-}" in --allow-unknown) ALLOW_UNKNOWN=1;; esac

MAX_WAIT="${MAX_WAIT_SECS:-300}"
[ "$MAX_WAIT" -le "$WINDOW" ] && MAX_WAIT=$(( WINDOW + 60 ))
START=$(date +%s)
IDLE_SINCE=""
POLL_WARNED=""

log() { echo "[idle-gate] $*" >&2; }

read -r -d '' JQ <<'JQEOF'
def pick($id):
  if $id == "" or $id == "auto"
  then ([.entries[]? | select(.status == "RUNNING" and (.telemetry.concurrent_requests != null))] | .[0])
  else ([.entries[]? | select(.id == $id)] | .[0])
  end;
pick($id) as $e
| if $e == null then ["missing", $id, "", "", ""]
  elif ($e.telemetry.concurrent_requests == null) or ($e.telemetry.queued_requests == null) then ["unknown", $e.id, "", "", ""]
  elif (($e.telemetry.concurrent_requests + $e.telemetry.queued_requests) > 0) then ["busy", $e.id, ($e.telemetry.concurrent_requests|tostring), ($e.telemetry.queued_requests|tostring), ($e.telemetry.live_tok_per_sec|tostring)]
  elif (($e.telemetry.live_tok_per_sec // 0) > 0.5) then ["busy", $e.id, "0", "0", ($e.telemetry.live_tok_per_sec|tostring)]
  else ["idle", $e.id, "0", "0", (($e.telemetry.live_tok_per_sec // 0)|tostring)]
  end | @tsv
JQEOF

while :; do
    NOW=$(date +%s)

    if [ -n "$IDLE_SINCE" ] && [ $(( NOW - IDLE_SINCE )) -ge "$WINDOW" ]; then
        log "OPEN: $RESOLVED idle for $(( NOW - IDLE_SINCE ))s (>= ${WINDOW}s) — safe to proceed"
        exit 0
    fi
    if [ $(( NOW - START )) -ge "$MAX_WAIT" ]; then
        log "TIMEOUT: $RESOLVED never stayed idle for ${WINDOW}s within ${MAX_WAIT}s (last sample: ${FLAG:-none} running=${CONC:-?} queued=${QUEUED:-?} tok/s=${TOK:-?}) — do not proceed"
        exit 1
    fi

    BODY=$(curl -fsS --max-time 4 "$URL/api/state" 2>/dev/null)
    if [ -z "$BODY" ]; then
        [ -z "$POLL_WARNED" ] && { log "poll failed ($URL) — resetting window"; POLL_WARNED=1; }
        IDLE_SINCE=""
        sleep "$SAMPLE"
        continue
    fi
    POLL_WARNED=""

    LINE=$(printf '%s' "$BODY" | jq -r --arg id "$ENTRY" "$JQ" 2>/dev/null)
    FLAG=""; RESOLVED="$ENTRY"; CONC=""; QUEUED=""; TOK=""
    if [ -n "$LINE" ]; then
        IFS=$'\t' read -r FLAG RESOLVED CONC QUEUED TOK <<<"$LINE"
    fi
    [ -z "$FLAG" ] && { FLAG="missing"; RESOLVED="$ENTRY"; }

    case "$FLAG" in
        idle)
            [ -z "$IDLE_SINCE" ] && { IDLE_SINCE=$NOW; log "$RESOLVED idle (0 running / 0 queued / 0 tok/s) — window started, need ${WINDOW}s"; }
            ;;
        unknown)
            if [ "$ALLOW_UNKNOWN" = 1 ]; then
                [ -z "$IDLE_SINCE" ] && { IDLE_SINCE=$NOW; log "$RESOLVED reports no telemetry, counting as idle (--allow-unknown)"; }
            else
                [ -n "$IDLE_SINCE" ] && log "$RESOLVED gauges unknown — window reset"
                IDLE_SINCE=""
            fi
            ;;
        *)
            [ -n "$IDLE_SINCE" ] && log "$RESOLVED $FLAG (running=$CONC queued=$QUEUED tok/s=$TOK) — window reset"
            IDLE_SINCE=""
            ;;
    esac

    sleep "$SAMPLE"
done
