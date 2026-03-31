#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" python -m open_infer.run_infer_json "$@"
