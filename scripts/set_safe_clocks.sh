#!/usr/bin/env bash
# Cap the Thor GPU compute clock to a stable operating point.
#
# WHY: On this dev kit, the GPU PCIe/power delivery is unstable at the top of the
# clock range (hard resets, no panic trace, PCIe AER RxErr on the GPU port) when
# the GPU is exercised at/near MAXN (~1341 MHz). A full sustained TTS benchmark
# hard-reset the board twice at stock clocks. Capping the GPU compute cluster
# (gpu-gpc-0) to 1100 MHz eliminated the resets across repeated sustained runs
# while keeping RTF well under the 0.5 target.
#
# Usage: sudo bash scripts/set_safe_clocks.sh [MHZ]   (default 1100)

set -euo pipefail
MHZ="${1:-1100}"
HZ=$((MHZ * 1000000))
NODE=/sys/class/devfreq/gpu-gpc-0

if [[ ! -d "$NODE" ]]; then
  echo "GPU devfreq node not found: $NODE" >&2
  exit 1
fi

echo "$HZ" > "$NODE/max_freq"
echo "GPU gpc max clock capped to ${MHZ} MHz (cur=$(cat "$NODE/cur_freq"))"
