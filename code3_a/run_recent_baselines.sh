#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python -u run_recent_paper_baselines.py --resume "$@"

