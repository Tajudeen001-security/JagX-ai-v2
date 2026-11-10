#!/usr/bin/env bash
# JagX v13 — end-to-end verification against a running server.
# Exercises the real HTTP surface: auth, SSE chat, tool calls, guard, audit
# trail, memory, goals, forge and evals.
#
#   bash scripts/e2e_check.sh
#   BASE=http://127.0.0.1:8000/api bash scripts/e2e_check.sh
set -uo pipefail

BASE="${BASE:-http://127.0.0.1:8000/api}"
EMAIL="e2e_$(date +%s)@example.com"
PASS="password123"
PASSED=0
FAILED=0

# Normalise JSON whitespace so a check works against both compact (FastAPI)
# and spaced (our SSE) encodings.
norm() { tr -d ' \n\t'; }

ok()   { echo "  ✓ $1"; PASSED=$((PASSED+1)); }
bad()  { echo "  ✗ $1"; FAILED=$((FAILED+1)); }
# has <label> <haystack> <needle>
has()  { if printf '%s' "$2" | norm | grep -qF "$(printf '%s' "$3" | norm)"; then ok "$1"; else bad "$1"; fi; }
# eq <label> <actual> <expected>
eq()   { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (got '$2')"; fi; }

echo "=== JagX v13 end-to-end verification ==="
echo

# --- health ---------------------------------------------------------------
echo "[1] Health & config"
H=$(curl -s "$BASE/health")
has "health ok" "$H" '"ok":true'
has "local-first inference" "$H" '"local_first":true'
has "no third-party key in use" "$H" 'no third-party API key'
C=$(curl -s "$BASE/config")
has "council coordinator present" "$C" '"Atlas"'
has "council critic present" "$C" '"Verity"'
has "guard hard boundary present" "$C" '"hard_boundary"'
has "council lanes present" "$C" '"scout"'

# --- auth -----------------------------------------------------------------
echo
echo "[2] Auth"
S=$(curl -s -X POST "$BASE/auth/signup" -H 'Content-Type: application/json' \
     -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}")
TOKEN=$(printf '%s' "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("token",""))' 2>/dev/null)
[ -n "$TOKEN" ] && ok "signup returns a token" || bad "signup returns a token"
AUTH="Authorization: Bearer $TOKEN"

ME=$(curl -s "$BASE/auth/me" -H "$AUTH")
has "authenticated /me" "$ME" "$EMAIL"

eq "unauthenticated request rejected" \
   "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/conversations")" "401"
eq "bad token rejected" \
   "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/conversations" -H 'Authorization: Bearer nonsense')" "401"

# --- chat with SSE --------------------------------------------------------
echo
echo "[3] Agent run (SSE) — tool call, guard, answer"
STREAM=$(curl -s -N -X POST "$BASE/chat" -H "$AUTH" -H 'Content-Type: application/json' \
         -d '{"message":"What is 17*23? Use the calculator.","mode":"loop"}')
has "conversation event" "$STREAM" '"type":"conversation"'
has "run_start event" "$STREAM" '"type":"run_start"'
has "tool_call event" "$STREAM" '"type":"tool_call"'
has "guard verdict event" "$STREAM" '"type":"guard"'
has "tool_result event" "$STREAM" '"type":"tool_result"'
has "answer event" "$STREAM" '"type":"answer"'
has "self-reflection event" "$STREAM" '"type":"reflection"'
has "calculator result reached the answer" "$STREAM" '391'

RUN_ID=$(printf '%s' "$STREAM" | grep '"type": "meta"' | head -1 \
         | python3 -c 'import sys,json;print(json.loads(sys.stdin.read().split("data: ",1)[1]).get("run_id",""))' 2>/dev/null)
CONV_ID=$(printf '%s' "$STREAM" | grep '"type": "conversation"' | head -1 \
         | python3 -c 'import sys,json;print(json.loads(sys.stdin.read().split("data: ",1)[1]).get("conversation_id",""))' 2>/dev/null)

# --- audit trail ----------------------------------------------------------
echo
echo "[4] Audit trail"
if [ -n "$RUN_ID" ]; then
  R=$(curl -s "$BASE/runs/$RUN_ID" -H "$AUTH")
  has "run persisted as done" "$R" '"status":"done"'
  has "tool step persisted" "$R" '"kind":"tool"'
  has "answer step persisted" "$R" '"kind":"answer"'
else
  bad "run id captured"; bad "tool step persisted"; bad "answer step persisted"
fi

# --- conversation persistence --------------------------------------------
echo
echo "[5] Conversation persistence"
if [ -n "$CONV_ID" ]; then
  D=$(curl -s "$BASE/conversations/$CONV_ID" -H "$AUTH")
  has "user message stored" "$D" '"role":"user"'
  has "assistant message stored" "$D" '"role":"assistant"'
else
  bad "conversation id captured"; bad "assistant message stored"
fi

# --- guard: blocked action ------------------------------------------------
echo
echo "[6] Guard blocks a credential action"
G=$(curl -s -X POST "$BASE/tools/run" -H "$AUTH" -H 'Content-Type: application/json' \
    -d '{"name":"fetch_url","arguments":{"url":"https://x.test/api_key"}}')
has "credential action blocked" "$G" '"verdict":"block"'

# --- guard: review queue --------------------------------------------------
echo
echo "[7] Guard holds a send for review"
curl -s -N -X POST "$BASE/chat" -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"message":"Send an email to the client about the invoice.","mode":"loop"}' > /dev/null
AP=$(curl -s "$BASE/approvals" -H "$AUTH")
has "approval queued" "$AP" '"status":"pending"'

# --- memory ---------------------------------------------------------------
echo
echo "[8] Memory"
M=$(curl -s -X POST "$BASE/memories" -H "$AUTH" -H 'Content-Type: application/json' \
    -d '{"content":"The user is building an autonomous agent called JagX.","layer":"semantic"}')
has "memory stored" "$M" '"layer":"semantic"'
ML=$(curl -s "$BASE/memories" -H "$AUTH")
has "memory listed" "$ML" 'JagX'
has "embedding backend reported" "$ML" '"embedding_backend"'
CO=$(curl -s -X POST "$BASE/memories/consolidate" -H "$AUTH")
has "consolidation ran" "$CO" '"promoted"'

# --- goals ----------------------------------------------------------------
echo
echo "[9] Goals"
GO=$(curl -s -X POST "$BASE/goals" -H "$AUTH" -H 'Content-Type: application/json' \
     -d '{"title":"Ship the JagX v13 release","success_criterion":"Repo pushed and tests green"}')
GID=$(printf '%s' "$GO" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
[ -n "$GID" ] && ok "goal created" || bad "goal created"
ADV=$(curl -s -N -X POST "$BASE/goals/$GID/advance" -H "$AUTH")
has "goal advanced with a checkpoint" "$ADV" '"type":"goal_checkpoint"'
GL=$(curl -s "$BASE/goals" -H "$AUTH")
has "goal listed" "$GL" 'Ship the JagX v13 release'

# --- forge ----------------------------------------------------------------
echo
echo "[10] Self-improvement (JagX Forge)"
FS=$(curl -s "$BASE/forge/stats" -H "$AUTH")
has "reflection loop on" "$FS" '"reflection":"on"'
has "experience replay on" "$FS" '"experience_replay":"on"'
has "weight-training loop reported" "$FS" '"weight_training"'
FL=$(curl -s "$BASE/forge/lessons" -H "$AUTH")
has "lessons recorded" "$FL" '"lessons"'
FV=$(curl -s "$BASE/forge/variants" -H "$AUTH")
has "prompt variants seeded" "$FV" '"task_class"'
OPT=$(curl -s -X POST "$BASE/forge/optimize" -H "$AUTH")
has "bandit optimisation ran" "$OPT" '"promoted"'
DS=$(curl -s "$BASE/forge/dataset" -H "$AUTH")
has "SFT dataset export works" "$DS" '"count"'
PR=$(curl -s "$BASE/forge/preferences" -H "$AUTH")
has "DPO preference export works" "$PR" '"count"'

# --- evals ----------------------------------------------------------------
echo
echo "[11] Eval harness"
EV=$(curl -s "$BASE/evals" -H "$AUTH")
EID=$(printf '%s' "$EV" | python3 -c 'import sys,json;d=json.load(sys.stdin)["evals"];print(d[0]["id"] if d else "")' 2>/dev/null)
[ -n "$EID" ] && ok "eval suites seeded" || bad "eval suites seeded"
if [ -n "$EID" ]; then
  ER=$(curl -s -X POST "$BASE/evals/$EID/run" -H "$AUTH")
  has "eval scored" "$ER" '"score"'
  CP=$(curl -s -X POST "$BASE/evals/$EID/compare" -H "$AUTH")
  has "variant comparison ran" "$CP" '"ranking"'
fi

# --- tools ----------------------------------------------------------------
echo
echo "[12] Tools & sandbox"
TL=$(curl -s "$BASE/tools" -H "$AUTH")
has "web_search registered" "$TL" '"web_search"'
has "run_code registered" "$TL" '"run_code"'
has "calculator registered" "$TL" '"calculator"'
has "local sandbox active" "$TL" '"local-subprocess"'
TR=$(curl -s -X POST "$BASE/tools/run" -H "$AUTH" -H 'Content-Type: application/json' \
     -d '{"name":"run_code","arguments":{"language":"python","code":"print(6*7)"}}')
has "sandbox executed code" "$TR" '42'
TC=$(curl -s -X POST "$BASE/tools/run" -H "$AUTH" -H 'Content-Type: application/json' \
     -d '{"name":"calculator","arguments":{"expression":"(1250*0.07)+89"}}')
has "calculator computed exactly" "$TC" '176.5'

# --- settings -------------------------------------------------------------
echo
echo "[13] Settings"
PS=$(curl -s -X PUT "$BASE/settings" -H "$AUTH" -H 'Content-Type: application/json' \
     -d '{"temperature":0.7,"max_steps":8}')
has "settings saved" "$PS" '"temperature":0.7'
has "settings persisted" "$(curl -s "$BASE/settings" -H "$AUTH")" '"max_steps":8'

echo
echo "=========================================="
printf "  PASSED: %-4s FAILED: %s\n" "$PASSED" "$FAILED"
echo "=========================================="
[ "$FAILED" -eq 0 ] && exit 0 || exit 1
