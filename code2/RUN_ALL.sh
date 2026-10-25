#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python run_dynamic_demand.py --mode full --workers 6
