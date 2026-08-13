#!/usr/bin/env bash
# Start a training run against a live fleet.
#
# Waits for two cards that clear vLLM's reservation rather than launching on
# whatever is freest: on a shared box, starting on a card that cannot hold the
# KV cache spends a full model load to fail at engine init, then does it again.
set -euo pipefail

ROOT="${NAV_ROOT:-/data/koe}"
REPO="${NAV_REPO:-$ROOT/simworld-nav}"
: "${EB_UE_ENDPOINTS:?set EB_UE_ENDPOINTS=http://<render-host>:18800}"

echo "== fleet check: $EB_UE_ENDPOINTS"
curl -sf -m 8 "$EB_UE_ENDPOINTS/healthz" | grep -q '"engine_connected": true' || {
  echo "fleet is not reachable or has no engine; start it on the render box first"
  exit 1
}
echo "   reachable"

need_free_mib="${ROLLOUT_MIN_FREE_MIB:-20500}"
echo "== waiting for 2 cards with >${need_free_mib} MiB free"
for _ in $(seq 1 120); do
  CARDS=$(nvidia-smi --query-gpu=index,memory.used,memory.total \
          --format=csv,noheader,nounits \
          | awk -F', ' -v m="$need_free_mib" '{f=$3-$2; if (f>m) print $1}' \
          | head -2 | paste -sd,)
  [ "$(echo "$CARDS" | tr ',' '\n' | grep -c .)" -ge 2 ] && break
  sleep 10
done
[ -n "${CARDS:-}" ] || { echo "no cards free"; exit 1; }
echo "   using $CARDS"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$CARDS" N_GPUS=2
export EB_LIVE_TELEMETRY_DIR="${EB_LIVE_TELEMETRY_DIR:-$ROOT/nav_telemetry}"
exec bash embodiedbench/training/vagen/train_grpo_embodied.sh
