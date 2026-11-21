from pathlib import Path
import subprocess,sys,json,os,fcntl,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parents[1]
os.environ.update(CUDA_VISIBLE_DEVICES=os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0')),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/jmri_mpl')
lock=open(R/'audit/continuation.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 d={'stage':stage,'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'physical_gpu':1,**kw};(R/'continuation_status.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(d,flush=True)
try:
 for script,label in [('fusion1059.py','nine_model_feature_extraction_and_fusion'),('prepare_roi9.py','nine_case_automatic_segmentation_and_roi'),('run.py','ROI_five_fold_training'),('finish1059.py','final_statistics')]:
  if os.environ.get('JMRI_RESUME_ROI')=='1' and script in ['fusion1059.py','prepare_roi9.py']:continue
  status(label)
  with open(R/'logs'/('stage_'+script+'.log'),'a') as f:subprocess.run([sys.executable,'-u',str(R/'code'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=R.parent.parent,check=True)
  if script=='prepare_roi9.py':assert (R/'inputs/roi_added_records.json').exists()
 assert (R/'COMPLETE.json').exists();status('complete')
except Exception:
 status('failed',error=traceback.format_exc());raise
