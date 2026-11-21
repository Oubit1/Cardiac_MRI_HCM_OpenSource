from pathlib import Path
import os,sys,json,gc,shutil
R=Path(__file__).resolve().parents[1];C=R.parent
os.chdir(C.parent);os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('JMRI_GPU_UUID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
sys.path.insert(0,str(C/'code'))
import common
common.GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES']
import train,torch,numpy as np,pandas as pd
sys.path.insert(0,str(C/'nested_validation/code'))
import fusion_stage
fusion_stage.ROOT=R
p=json.loads((R/'plans/protocol.json').read_text());p['fusion']=json.loads((C/'nested_validation/plans/protocol.json').read_text())['fusion']
# Do not change the training protocol hash; fusion reads a separate temporary ROOT.
F=R/'roi_fusion';(F/'plans').mkdir(parents=True,exist_ok=True);(F/'plans/protocol.json').write_text(json.dumps(p));fusion_stage.ROOT=F
records=json.loads((R/'inputs/prepared_patients.json').read_text())
added={int(a['ID']):a for a in json.loads((R/'inputs/roi_added_records.json').read_text())}
for a in records:
 if a['cohort']=='development' and int(a['ID']) in added:a['roi_paths']=added[int(a['ID'])]['roi_paths']
dev={int(a['ID']):i for i,a in enumerate(records) if a['cohort']=='development'};ext={int(a['ID']):i for i,a in enumerate(records) if a['cohort']=='external_test'}
@torch.inference_mode()
def extract(model,ids,lookup,path,frozen=False):
 if path.exists():
  with np.load(path) as z:assert np.array_equal(z['ID'],ids)
  return
 ds=train.Patients(records,[lookup[int(i)] for i in ids],'roi',False);vals=[]
 for j in range(len(ds)):
  video=ds[j][0].cuda()
  with torch.autocast('cuda',dtype=torch.float16):
   encoded=model.encode(video)
   if frozen:z=torch.nn.functional.normalize(encoded.float().mean(0),dim=0)
   else:z,_=model.aggregator(encoded)
  vals.append(z.float().cpu().numpy());del video
  if j%100==0:print(path.name,j,len(ids),flush=True)
 with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,ID=np.array(ids),features=np.stack(vals))
 path.with_suffix('.tmp').replace(path)
source=C/'nested_validation/tabular_full/all_features'
folds=json.loads((R/'plans/outer_folds.json').read_text())
for k in range(1,7):
 final=k==6;old=source/('final' if final else f'fold_{k}/outer');dest=F/('final' if final else f'fold_{k}');dest.mkdir(exist_ok=True)
 for name in ['training.csv','prediction_features.csv','rscore_training_crossfit.csv','rscore_prediction.csv']:
  if not (dest/name).exists():shutil.copy2(old/name,dest/name)
 if not (dest/'cmr_selection').exists():shutil.copytree(old/'cmr_selection',dest/'cmr_selection')
 tr=pd.read_csv(dest/'training.csv').ID.astype(int).tolist();te=pd.read_csv(dest/'prediction_features.csv').ID.astype(int).tolist()
 weight=R/'runs'/('final_roi' if final else f'oof_roi_fold{k}')/'final.pt'
 cfg=json.loads((weight.parent/'config.json').read_text());assert set(cfg['train_ids'])==set(tr) and not cfg['validation_ids']
 (dest/'split.json').write_text(json.dumps({'outer_train':tr,'outer_test':te,'image_checkpoint':str(weight)}))
 if (dest/'fusion_complete.json').exists():continue
 print('FEATURES',k,flush=True)
 model=train.make_model(8);model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=False)['model']);model.eval()
 extract(model,tr,dev,dest/'image_train.npz');extract(model,te,ext if final else dev,dest/'image_heldout.npz');del model;gc.collect();torch.cuda.empty_cache()
 # Same outcome-independent encoder, mean pooling and L2 normalization for every patient.
 cache=R/'fusion/frozen';cache.mkdir(exist_ok=True)
 if not (cache/'development.npz').exists() or not (cache/'external.npz').exists():
  model=train.make_model(8);model.eval()
  extract(model,list(dev),dev,cache/'development.npz',True);extract(model,list(ext),ext,cache/'external.npz',True)
  del model;gc.collect();torch.cuda.empty_cache()
 for role,ids,co in [('train',tr,'development'),('heldout',te,'external' if final else 'development')]:
  with np.load(cache/(co+'.npz')) as z:frame=pd.DataFrame(z['features'],index=z['ID'])
  np.savez_compressed(dest/f'frozen_{role}.npz',ID=np.array(ids),features=frame.loc[ids].values)
 fusion_stage.run_fusion(dest,final=final)
 print('FUSION COMPLETE',k,flush=True)
oof=pd.concat([pd.read_csv(F/f'fold_{k}/predictions.csv') for k in range(1,6)])
truth=pd.DataFrame([{'ID':a['ID'],'event':a['event']} for a in records if a['cohort']=='development'])
oof=oof.merge(truth,on='ID',validate='one_to_one');assert len(oof)==1059
oof.to_csv(R/'results/roi_fusion_oof_predictions.csv',index=False)
y=pd.DataFrame([{'ID':a['ID'],'event':a['event']} for a in records if a['cohort']=='external_test'])
e=pd.read_csv(F/'final/predictions.csv').merge(y,on='ID',validate='one_to_one');assert len(e)==309
e.to_csv(R/'results/roi_fusion_external_predictions.csv',index=False)
(R/'results/roi_fusion_complete.json').write_text(json.dumps({'development_n':1059,'external_n':309,'models':fusion_stage.MODELS}))
