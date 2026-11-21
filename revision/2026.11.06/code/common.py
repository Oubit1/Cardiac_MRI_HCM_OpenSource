from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATASET = Path(os.environ.get('JMRI_DATASET', str(PROJECT/'data')))
GPU_UUID = os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def status(stage, **kwargs):
    record = {'updated_at': timestamp(), 'stage': stage, **kwargs}
    write_json(ROOT / 'status.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def pin_gpu():
    existing = os.environ.get('CUDA_VISIBLE_DEVICES')
    if existing not in (None, GPU_UUID):
        raise RuntimeError(f'Unexpected CUDA_VISIBLE_DEVICES={existing}; refusing another GPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
