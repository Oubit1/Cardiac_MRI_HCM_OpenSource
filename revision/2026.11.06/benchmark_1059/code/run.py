from pathlib import Path
import os,sys,json,fcntl,traceback
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
os.chdir(CORE.parent)
GPU=os.environ.get('JMRI_GPU_UUID',os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0')))
worker=os.environ.get('JMRI_WORKER_FOLD')
physical_gpu=int(os.environ.get('JMRI_PHYSICAL_GPU','1'))
os.environ['CUDA_VISIBLE_DEVICES']=GPU
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/jmri_mpl'
sys.path.insert(0,str(CORE/'code'))
import common
common.GPU_UUID=GPU
import train
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
train.ROOT=ROOT
lock=(ROOT/('audit/worker_'+worker+'.lock' if worker else 'audit/queue.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 common.write_json(ROOT/(f'status_fold{worker}.json' if worker else 'status.json'),dict(stage=stage,time=common.timestamp(),physical_gpu=physical_gpu,**kw));print(stage,kw,flush=True)
train.status=status

original_train_job=train.train_job
original_infer_job=train.infer_job
def locked_train(*args,**kwargs):
 directory=Path(args[4]);directory.mkdir(parents=True,exist_ok=True)
 with (directory/'training.lock').open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  return original_train_job(*args,**kwargs)
def locked_infer(*args,**kwargs):
 path=Path(args[4]);path.parent.mkdir(parents=True,exist_ok=True)
 with (path.parent/'inference.lock').open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  if path.exists():return
  return original_infer_job(*args,**kwargs)
train.train_job=locked_train
train.infer_job=locked_infer

records=json.loads((ROOT/'inputs/prepared_patients.json').read_text())
lookup={int(r['ID']):i for i,r in enumerate(records) if r['cohort']=='development'}
ext=[i for i,r in enumerate(records) if r['cohort']=='external_test']
assert len(lookup)==1059 and len(ext)==309
folds=json.loads((ROOT/'plans/outer_folds.json').read_text())
def cfg(n,seed,mode):return dict(encoder_lr=1e-5,epochs=int(n),seed=seed,mode=mode,temporal='full',microbatch=8)
try:
 for mode in (['roi'] if worker else ['full','roi']):
  if mode=='roi':
   patch=ROOT/'inputs/roi_added_records.json'
   if not patch.exists():
    status('full_training_complete_roi_preparation_required',note='Full mViT training done; ROI and nine-model fusion are not complete.');sys.exit(0)
   additions={int(r['ID']):r for r in json.loads(patch.read_text())}
   for r in records:
    if r['cohort']=='development' and int(r['ID']) in additions:r['roi_paths']=additions[int(r['ID'])]['roi_paths']
  for r in records:
   paths=r['full_paths' if mode=='full' else 'roi_paths'];assert paths and all(Path(p).is_file() for p in paths)
  selections=[]
  for sp in folds:
   if worker and int(sp['split'])!=int(worker):continue
   f=int(sp['split']);outer=[lookup[int(i)] for i in sp['training_ids']];held=[lookup[int(i)] for i in sp['heldout_ids']]
   tr,va=train_test_split(outer,test_size=.1,random_state=42+f,stratify=[records[i]['event'] for i in outer])
   assert not set(outer)&set(held)
   common.write_json(ROOT/f'plans/fold_{f}.json',{k:[records[i]['ID'] for i in v] for k,v in dict(outer_train=outer,outer_test=held,inner_train=tr,inner_validation=va).items()})
   status('training',mode=mode,fold=f,development_n=1059,outer_train_n=len(outer),outer_test_n=len(held))
   result=train.train_job(records,tr,va,cfg(60,42+f,mode),ROOT/f'runs/inner_{mode}_fold{f}')
   epoch=int(result['best']['60']['epoch']);selections.append(dict(fold=f,epoch=epoch))
   common.write_json(ROOT/f'results/{mode}_selected_epochs.json',selections)
   dest=ROOT/f'runs/oof_{mode}_fold{f}'
   train.train_job(records,outer,None,cfg(epoch,42+f,mode),dest)
   train.infer_job(records,held,cfg(epoch,42+f,mode),dest/'final.pt',dest/'heldout_predictions.csv')
  if worker:
   status('worker_complete',fold=int(worker));sys.exit(0)
  pd.concat([pd.read_csv(ROOT/f'runs/oof_{mode}_fold{f}/heldout_predictions.csv') for f in range(1,6)]).to_csv(ROOT/f'results/{mode}_oof_predictions.csv',index=False)
  epoch=int(np.median([r['epoch'] for r in selections]));dest=ROOT/f'runs/final_{mode}'
  train.train_job(records,list(lookup.values()),None,cfg(epoch,42,mode),dest)
  train.infer_job(records,ext,cfg(epoch,42,mode),dest/'final.pt',dest/'external_predictions.csv')
  common.write_json(ROOT/f'results/{mode}_training_complete.json',{'development_n':1059,'external_n':309,'final_epochs':epoch})
 status('mvit_training_complete_fusion_pending')
except Exception:
 status('failed',traceback=traceback.format_exc());raise
