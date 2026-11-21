from pathlib import Path
import sys,os,json,fcntl,traceback,shutil
FOLD=int(sys.argv[1]);assert FOLD in [3,4,5]
MAIN=Path(__file__).resolve().parents[1];CORE=MAIN.parent;ROOT=MAIN/f'parallel/fold{FOLD}'
os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'));os.environ['MPLCONFIGDIR']=str(ROOT/'mpl');os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0,str(CORE/'code'))
import common
common.GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES']
import train
train.ROOT=ROOT
lock=(ROOT/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 if 'gpu' in kw:kw['gpu']='GPU 3'
 common.write_json(ROOT/'status.json',{'stage':stage,'fold':FOLD,'gpu':3,'updated_at':common.timestamp(),**kw});print(stage,kw,flush=True)
train.status=status
records=json.loads((CORE/'inputs/prepared_patients.json').read_text());lookup={int(r['ID']):i for i,r in enumerate(records) if r['cohort']=='development'}
split=json.loads((CORE/f'plans/fold_{FOLD}_ids.json').read_text());tr,va,outer,held=([lookup[int(i)] for i in split[k]] for k in ['inner_train','inner_validation','outer_train','outer_test'])
assert not set(tr)&set(va) and set(tr)|set(va)==set(outer) and not set(outer)&set(held)
def cfg(e):return dict(encoder_lr=1e-5,epochs=e,seed=42+FOLD,mode='roi',temporal='full',microbatch=8)
try:
 result=train.train_job(records,tr,va,cfg(60),ROOT/f'runs/inner_roi_fold{FOLD}')
 epoch=int(result['best']['60']['epoch']);assert 1<=epoch<=60
 dest=ROOT/f'runs/oof_roi_fold{FOLD}';train.train_job(records,outer,None,cfg(epoch),dest)
 train.infer_job(records,held,cfg(epoch),dest/'final.pt',dest/'heldout_predictions.csv')
 # Main coordinator is held at the fold-2 report barrier until publication completes.
 for name in [f'inner_roi_fold{FOLD}',f'oof_roi_fold{FOLD}']:
  origin=ROOT/'runs'/name;target=MAIN/'runs'/name
  assert not target.exists(),f'Unexpected concurrent target: {target}'
  origin.rename(target)
 common.write_json(MAIN/f'audit/parallel_fold{FOLD}_complete.json',{'fold':FOLD,'epochs':epoch,'gpu':3,'complete':True})
 status('complete',selected_epoch=epoch)
except Exception:
 status('failed',traceback=traceback.format_exc());raise
