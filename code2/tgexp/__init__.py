"""New multitask experiments on unchanged ServiceOpt-v3 physical components."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor_service_v3'
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

