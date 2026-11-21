from pathlib import Path
import os,sys,json,gc,shutil,fcntl,traceback,copy,zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parents[1];C=R.parent;OUT=R/'roi_frozen';OUT.mkdir(exist_ok=True)
lock=open(OUT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(stage,**kw):
 obj={'stage':stage,'time':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'gpu':1,**kw};(OUT/'status.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2));print(obj,flush=True)
try:
 # Reuse the tested input loader and extractor, but a distinct frozen-ROI cache.
 code=(R/'code/roi_fusion1059.py').read_text().split("source=C/'nested_validation/tabular_full/all_features'")[0]
 code=code.replace("F=R/'roi_fusion'","F=R/'roi_frozen'")
 exec(compile(code,str(R/'code/roi_fusion1059.py'),'exec'),globals())
 fusion_stage.MODELS=['Frozen','Frozen_CMR','Frozen_Rscore_CMR']
 status('extracting_frozen_roi_features')
 model=train.make_model(8);model.eval()
 for par in model.parameters():par.requires_grad_(False)
 cache=OUT/'features';cache.mkdir(exist_ok=True)
 extract(model,list(dev),dev,cache/'development.npz',True)
 extract(model,list(ext),ext,cache/'external.npz',True)
 del model;gc.collect();torch.cuda.empty_cache()
 common.write_json(OUT/'protocol.json',{'development_n':1059,'external_n':309,'encoder':'unchanged pretrained.ckpt','pretrained_sha256':common.sha256(C/'inputs/pretrained.ckpt'),'pooling':'mean over slices then L2 normalization','supervised_encoder_training':False,'downstream':p['fusion'],'input':'same ROI caches as ROI-FT','outer_folds':str(R/'plans/outer_folds.json')})
 for k in range(1,7):
  final=k==6;name='final' if final else f'fold_{k}';old=R/'roi_fusion'/name;dest=OUT/name;dest.mkdir(exist_ok=True)
  for file in ['training.csv','prediction_features.csv','rscore_training_crossfit.csv','rscore_prediction.csv','split.json','image_train.npz','image_heldout.npz']:
   if not (dest/file).exists():shutil.copy2(old/file,dest/file)
  if not (dest/'cmr_selection').exists():shutil.copytree(old/'cmr_selection',dest/'cmr_selection')
  tr=pd.read_csv(dest/'training.csv').ID.tolist();te=pd.read_csv(dest/'prediction_features.csv').ID.tolist()
  for role,ids,co in [('train',tr,'development'),('heldout',te,'external' if final else 'development')]:
   with np.load(cache/(co+'.npz')) as z:frame=pd.DataFrame(z['features'],index=z['ID'])
   assert np.allclose(np.linalg.norm(frame.values,axis=1),1,atol=1e-4)
   np.savez_compressed(dest/f'frozen_{role}.npz',ID=np.array(ids),features=frame.loc[ids].values)
  status('fusion',fold=name);fusion_stage.run_fusion(dest,final=final)
 a=pd.concat([pd.read_csv(OUT/f'fold_{k}/predictions.csv') for k in range(1,6)])
 truth=pd.DataFrame([{'ID':v['ID'],'event':v['event']} for v in records if v['cohort']=='development']);a=a.merge(truth,on='ID',validate='one_to_one');assert len(a)==1059
 b=pd.read_csv(OUT/'final/predictions.csv');truth=pd.DataFrame([{'ID':v['ID'],'event':v['event']} for v in records if v['cohort']=='external_test']);b=b.merge(truth,on='ID',validate='one_to_one');assert len(b)==309
 a.to_csv(OUT/'oof_predictions.csv',index=False);b.to_csv(OUT/'external_predictions.csv',index=False)
 from revision_statistics import metrics_ci
 from sklearn.metrics import roc_curve
 stats=[];status('bootstrap_statistics')
 for m in fusion_stage.MODELS:
  fp,tp,thr=roc_curve(a.event,a[m]);ok=np.where((tp>=.95)&np.isfinite(thr))[0];cut=float(thr[ok[np.argmax((1-fp)[ok])]])
  for co,d in [('development',a),('external',b)]:stats.append({'model':m,'cohort':co,**metrics_ci(d.event,d[m],cut=cut,rounds=2000)})
 x=pd.DataFrame(stats);x.to_csv(OUT/'performance.csv',index=False)
 with pd.ExcelWriter(OUT/'ROI_Frozen_1059完整结果.xlsx') as wr:
  x.to_excel(wr,sheet_name='Performance',index=False);a.to_excel(wr,sheet_name='OOF',index=False);b.to_excel(wr,sheet_name='External',index=False)
 status('complete');(OUT/'COMPLETE.json').write_text(json.dumps({'complete':True,'models':fusion_stage.MODELS,'n_development':1059,'n_external':309}))

except Exception:
 status('failed',traceback=traceback.format_exc());raise
