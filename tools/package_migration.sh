#!/bin/bash
# Bundle everything the new machine needs that git does not carry.
#
# Two things git cannot bring: vendor/ is ignored (3.6 GB, of which the courier
# path needs 6.4 MB of map JSON) and the baked albums live outside the repo.
# Everything else comes from the clone.
set -euo pipefail
OUT="${1:-/data/murray/courier-migration}"
MAPS="$(git rev-parse --show-toplevel)/vendor/vagen/vagen/envs/deliverybench/maps"
mkdir -p "$OUT"

echo "maps (6.4 MB) ..."
tar -C "$(dirname "$MAPS")" -czf "$OUT/maps.tar.gz" maps

echo "albums (~2 GB) ..."
tar -C /data/murray -czf "$OUT/albums.tar.gz" \
    paris_streets_v2 paris_lamps_real paris_obstacles

echo "measurements and recorded runs ..."
tar -C /data/murray -czf "$OUT/records.tar.gz" \
    --ignore-failed-read \
    traj_by_tier.json stride_sweep.json paris_signal_rows.jsonl \
    capture_trajectories.py sweep_strides.py 2>/dev/null || true

( cd "$OUT" && sha256sum ./*.tar.gz > SHA256SUMS )
cat > "$OUT/README" <<TXT
Unpack on the new machine:

  git clone <remote> simworld_nav && cd simworld_nav
  git checkout courier-environment
  sha256sum -c $OUT/SHA256SUMS
  tar -C vendor/vagen/vagen/envs/deliverybench -xzf maps.tar.gz
  tar -C /data/murray -xzf albums.tar.gz
  tar -C /data/murray -xzf records.tar.gz

Then verify:

  python -m pytest tests/ -q                     # 950 passed, 4 skipped
  python -m embodiedbench.tools.migration_check  # exit 0

See docs/MIGRATION.md. The Unreal Engine tree is NOT in here on purpose --
it is a build tool for making new photographs, not part of the environment.
TXT
du -sh "$OUT"/*.tar.gz
echo "wrote $OUT"
