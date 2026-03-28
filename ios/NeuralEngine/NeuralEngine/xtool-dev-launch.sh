#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVX_BIN="${UVX_BIN:-/home/ales27pm/.local/bin/uvx}"
BUNDLE_ID="${BUNDLE_ID:-com.27pm.neural}"
UDID="${UDID:-${1:-}}"
RSD_HOST="${XTL_RSD_HOST:-${2:-}}"
RSD_PORT="${XTL_RSD_PORT:-${3:-}}"

if [[ -z "${UDID}" ]]; then
  echo "usage: $(basename "$0") <udid> [rsd-host] [rsd-port]" >&2
  exit 2
fi

if [[ ! -x "${UVX_BIN}" ]]; then
  echo "uvx not found at ${UVX_BIN}" >&2
  exit 1
fi

(
  cd "${ROOT_DIR}"
  xtool dev run --udid "${UDID}"
)

if [[ -z "${RSD_HOST}" || -z "${RSD_PORT}" ]]; then
  cat >&2 <<EOF
xtool install completed.
To launch the developer build, rerun with an active CoreDevice tunnel:
  XTL_RSD_HOST=<host> XTL_RSD_PORT=<port> $(basename "$0") ${UDID}
EOF
  exit 0
fi

RESOLVED_BUNDLE_ID="$(
  "${UVX_BIN}" pymobiledevice3 apps list --udid "${UDID}" | python3 - "${BUNDLE_ID}" <<'PY'
import json
import sys

apps = json.load(sys.stdin)
base = sys.argv[1]

developer = sorted(
    key for key in apps
    if key.startswith("XTL-") and key.endswith("." + base)
)

print(developer[-1] if developer else base)
PY
)"

"${UVX_BIN}" pymobiledevice3 developer core-device launch-application \
  --rsd "${RSD_HOST}" "${RSD_PORT}" \
  "${RESOLVED_BUNDLE_ID}" ""
