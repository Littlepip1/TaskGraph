from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protocol():
    return load_json(ROOT / 'configs' / 'final_airsim_protocol.json')


def base_config():
    return load_json(ROOT / 'configs' / 'airsim_final_n64.json')


def atomic_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def source_hash():
    h = hashlib.sha256()
    paths = []
    for rel in ['airsim_ext','r4','r6f','r7','r8a','r8b','r8_airsim_final','final_airsim','tests']:
        paths.extend((ROOT / rel).rglob('*.py'))
    paths.extend((ROOT / 'configs').glob('*.json'))
    paths.extend(ROOT.glob('*.py'))
    paths.extend(ROOT.glob('*.json'))
    for f in sorted(set(paths)):
        if any(part in {'runs','__pycache__'} for part in f.relative_to(ROOT).parts):
            continue
        h.update(f.relative_to(ROOT).as_posix().encode('utf-8'))
        h.update(f.read_bytes())
    return h.hexdigest()


def runtime(allow_non_target=False):
    target = sys.version_info[:3] == (3, 11, 13)
    if not target and not allow_non_target:
        raise RuntimeError('Use the existing Python 3.11.13 Anaconda environment; do not reinstall or upgrade Python.')
    import numpy
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = None
    return {
        'python': platform.python_version(),
        'numpy': numpy.__version__,
        'scipy': scipy_v,
        'platform': platform.platform(),
        'target_python': target,
    }
