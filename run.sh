#!/usr/bin/env bash
# Run generate.py in a container on whichever GPU this machine has.
#
#   ./run.sh --prompt "A red fox trotting through fresh snow at sunrise"
#   TEXT2VIDEO_BACKEND=xpu ./run.sh --prompt "..."     # force a backend instead of detecting it
#
# Detection: an NVIDIA driver (nvidia-smi lists a GPU) selects the CUDA image; otherwise an Intel GPU device node
# (/dev/dri, or /dev/dxg under WSL2) selects the XPU image. Everything after the script name goes to generate.py.
set -euo pipefail
cd "$(dirname "$0")"

detect_backend() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    echo cuda
  elif [ -e /dev/dri ] || [ -e /dev/dxg ]; then
    echo xpu
  else
    echo "No NVIDIA or Intel GPU found (no nvidia-smi GPU, no /dev/dri, no /dev/dxg)." >&2
    exit 1
  fi
}

backend="${TEXT2VIDEO_BACKEND:-$(detect_backend)}"
case "$backend" in
  cuda|xpu) ;;
  *) echo "TEXT2VIDEO_BACKEND must be cuda or xpu, got '$backend'" >&2; exit 1 ;;
esac

echo "Backend: $backend" >&2
mkdir -p models outputs
exec docker compose --profile "$backend" run --rm "text2video-$backend" "$@"
