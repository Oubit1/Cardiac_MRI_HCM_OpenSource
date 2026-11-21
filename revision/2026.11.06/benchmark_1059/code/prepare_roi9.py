from pathlib import Path
import os,sys,json,subprocess,hashlib
import numpy as np,pandas as pd,nibabel as nib,pydicom
from scipy import ndimage
R=Path(__file__).resolve().parents[1];C=R.parent
os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'));os.environ['MPLCONFIGDIR']='/tmp/jmri_mpl'
sys.path.insert(0,str(C.parent/'revision_core_package/code/hcm_foundation_benchmark'))
import prepare_sax_nifti as cv
cv.selected_files=lambda p,u:(f for f in p.rglob('*') if f.is_file() and f.name not in {'VERSION','LOCKFILE','DICOMDIR'})
sys.path.insert(0,str(C/'code'));from prepare_inputs import resize,save_array
rows=pd.read_csv(C/'nine_case_impact/results/slices.csv');added=json.loads((C/'nine_case_impact/cache/prepared_nine.json').read_text());raw=R/'roi9/images';raw.mkdir(parents=True,exist_ok=True)
for rec in added:
 ident=int(rec['ID']);case=C/f'inputs/additional_images_2026-11-10/9例补图/短轴cine序列/{ident}';uids=set()
 for f in case.rglob('*'):
  if not f.is_file() or f.name in {'VERSION','LOCKFILE','DICOMDIR'}:continue
  try:
   d=pydicom.dcmread(f,stop_before_pixels=True)
   if d.get('Rows'):uids.add(str(d.SeriesInstanceUID))
  except Exception:pass
 groups,_,_=cv.collect_cine_groups(case,uids,12);groups=cv.choose_unique_layers(groups)
 for j,(_,row) in enumerate(rows[rows.ID==ident].iterrows()):
  dest=raw/f'{ident}_slice{j+1:02}.nii.gz'
  if dest.exists():continue
  g=groups[int(row.slice_index)-1];assert str(g['frame_paths'][0])==row.first_dicom
  a=cv.load_group(g);assert a.shape==(row.frames,row.height,row.width)
  lo,hi=np.percentile(a,[1,99]);norm=(np.clip(a,lo,hi)-lo)/(hi-lo)
  assert np.allclose(resize(norm),np.load(row.cache),atol=1e-5)
  nib.save(nib.Nifti1Image(a.astype('float32'),np.eye(4)),dest)
seg=Path(os.environ.get('JMRI_SEGMENTATION_ROOT', 'Automated_Annotation_of_Cine'));weight=seg/'runs/unet2d_myo_all_centers/weights/best_model.pt'
assert weight.is_file()
receipt={'model':str(weight),'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'threshold':.5,'labels':'automatic predictions, not manual ground truth','scope':'nine added development patients only'}
(R/'roi9/model_source.json').write_text(json.dumps(receipt,indent=2))
subprocess.run([sys.executable,str(seg/'main_multicenter.py'),'predict','--weights',str(weight),'--input',str(raw),'--output-dir',str(R/'roi9/annotation'),'--device','cuda','--visualize'],check=True,cwd=seg)
audit=[]
for rec in added:
 ident=int(rec['ID']);paths=[]
 for j,full in enumerate(rec['full_paths']):
  name=f'{ident}_slice{j+1:02}.nii.gz';a=nib.load(raw/name).get_fdata().astype('float32');m=nib.load(R/'roi9/annotation/masks'/name).get_fdata()>0;assert a.shape==m.shape
  area=m.sum((1,2));fraction=float(np.median(m.mean((1,2))));components=float(np.median([ndimage.label(v)[1] for v in m]));den=area[:-1]+area[1:];dice=float(np.median((2*np.logical_and(m[:-1],m[1:]).sum((1,2))+1)/(den+1)))
  fail=(area==0).any() or not .002<=fraction<=.30 or components>2 or dice<.5
  target=R/f'roi9/cache/{ident}_slice{j+1:02}.npy';bbox=None
  if fail:out=full
  else:
   yy,xx=np.where(m.any(0));h,w=a.shape[-2:];cy,cx=(yy.min()+yy.max()+1)/2,(xx.min()+xx.max()+1)/2;side=min(int(np.ceil(max(yy.max()-yy.min()+1,xx.max()-xx.min()+1,32)*1.4)),h,w);top=int(np.clip(round(cy-side/2),0,h-side));left=int(np.clip(round(cx-side/2),0,w-side));bbox=[top,left,side,side]
   lo,hi=np.percentile(a,[1,99]);norm=(np.clip(a,lo,hi)-lo)/max(hi-lo,1e-6);save_array(target,resize(norm[:,top:top+side,left:left+side]));out=str(target)
  paths.append(out);audit.append({'ID':ident,'slice':j+1,'fallback_full':bool(fail),'fraction':fraction,'components':components,'temporal_dice':dice,'bbox':json.dumps(bbox),'mask':str(R/'roi9/annotation/masks'/name)})
 rec['roi_paths']=paths
pd.DataFrame(audit).to_csv(R/'roi9/issue_manifest.csv',index=False)
(R/'roi9/summary.json').write_text(json.dumps({'patients':9,'slices':len(audit),'fallback_slices':sum(a['fallback_full'] for a in audit),'policy':'unchanged original ROI QC and fallback; no mask repair'},indent=2))
(R/'inputs/roi_added_records.json').write_text(json.dumps(added,ensure_ascii=False,indent=2));print('ROI READY',len(added),flush=True)
