from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,os,sys,shutil
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'audit/matplotlib'))

ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT.parent
PROJECT=CORE.parent
PYTHON=sys.executable
RSCRIPT=os.environ.get('JMRI_RSCRIPT', shutil.which('Rscript') or 'Rscript')
GPU_UUID=os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
MODELS=['CMR','Rscore','Rscore_CMR','mViT','mViT_CMR','mViT_Rscore_CMR','Frozen','Frozen_CMR','Frozen_Rscore_CMR']

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def status(stage,**kw):
    obj={'time':now(),'stage':stage,**kw};write_json(ROOT/'status.json',obj);print(json.dumps(obj),flush=True)
