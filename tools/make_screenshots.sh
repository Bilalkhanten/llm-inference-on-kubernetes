#!/usr/bin/env bash
# Re-runs every tool and renders the output into docs/img/*.png.
# The PNGs are renders of REAL output; nothing in them is typed by hand.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
K8S=${K8S_VERSION:-1.34}
OLD_K8S=${OLD_K8S_VERSION:-1.33}
TITLE="terminal: llm-inference-on-kubernetes"
mkdir -p out docs/img

$PY tools/kvcalc.py --ctx 8192 > out/kvcalc.txt
$PY tools/captool.py --qps 60 --e2e 6 --ramp-qps-per-min 20 > out/captool.txt
$PY tools/routesim.py --lam 80 > out/routesim80.txt
$PY tools/validate.py "$K8S" > out/validate.txt
$PY tools/validate.py "$OLD_K8S" > out/validate_old.txt || true

{ echo '$ python3 tools/kvcalc.py --ctx 8192'; cat out/kvcalc.txt; } \
  | $PY tools/render.py docs/img/ss1-kv-capacity.png "$TITLE"

{ echo '$ python3 tools/captool.py --qps 60 --e2e 6 --ramp-qps-per-min 20'; cat out/captool.txt; } \
  | $PY tools/render.py docs/img/ss2-capacity-headroom.png "$TITLE"

# routesim output layout: 2 header lines, blank, section A (7 lines), blank, section B
{ echo '$ python3 tools/routesim.py --lam 80   # section A'; sed -n '1,10p' out/routesim80.txt; } \
  | $PY tools/render.py docs/img/ss3a-routing-mild-skew.png "$TITLE"

{ echo '$ python3 tools/routesim.py --lam 80   # section B'; sed -n '1,2p' out/routesim80.txt; sed -n '11,100p' out/routesim80.txt; } \
  | $PY tools/render.py docs/img/ss3b-routing-hot-prefix.png "$TITLE"

{ echo "\$ python3 tools/validate.py $K8S"; cat out/validate.txt; echo; \
  echo "\$ python3 tools/validate.py $OLD_K8S | grep ResourceClaimTemplate"; grep ResourceClaimTemplate out/validate_old.txt; } \
  | $PY tools/render.py docs/img/ss4-offline-schema-validation.png "$TITLE"

echo "wrote docs/img/ss1..ss4 from real tool output (raw text is in out/)"
