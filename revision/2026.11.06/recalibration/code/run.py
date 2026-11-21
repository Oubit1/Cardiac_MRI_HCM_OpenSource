from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import log_loss,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent;SOURCE=CORE/'nested_validation/tabular_full/all_features'
sys.path.insert(0,str(CORE/'nested_validation/code'))
from tabular_suite import matrices,estimator
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import metrics_ci,calibration
MODELS={'Rscore_CMR_CatBoost':'CatBoost','Rscore_CMR_EN':'EN'}
def write(name,obj): (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def fit(y,p):
 z=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y)
 def fn(b):
  eta=b[0]+b[1]*z;res=expit(eta)-y
  return np.mean(np.logaddexp(0,eta)-y*eta),np.array([res.mean(),(res*z).mean()])
 r=minimize(fn,[0,1],jac=True,method='L-BFGS-B',bounds=[(None,None),(1e-6,None)],options={'ftol':1e-13,'gtol':1e-9,'maxiter':1000})
 if not r.success:raise RuntimeError(r.message)
 return {'intercept':float(r.x[0]),'slope':float(r.x[1]),'positive_slope_boundary':bool(r.x[1]<=1.01e-6),'optimizer_success':True}
def apply(p,c):return expit(c['intercept']+c['slope']*logit(np.clip(p,1e-6,1-1e-6)))
def main():
 protocol={'date':'2026-11-10','primary':'Rscore_CMR_CatBoost','secondary':'Rscore_CMR_EN','method':'Unweighted logistic calibration of clipped logit, free intercept and positive slope; no calibration method search','outer_evaluation':'Existing five outer partitions. For each, rebuild downstream classifiers on saved inner-training upstream matrices, collect held-out inner predictions, fit calibrator only in outer training, transform saved outer-test predictions.','reuse':'CMR/radiomics selection and Rscore crossfits from the existing inner runs; no global score reused in inner splits.','hyperparameters':'Previously selected outer-training CatBoost parameters reused across its inner folds; inner predictions are cross-fitted conditional on this selection, not separately nested for hyperparameter choice. Outer-test patients remain excluded from all calibration and selection.','final':'Fit calibrator to development outer OOF probabilities and outcomes, apply to saved full-development model external predictions. Training-size/selected-parameter differences may affect calibration transfer.','external_reused':True,'external_labels_for_fitting':False,'bootstrap':2000,'operating_points':'Preserve decisions by mapping original locked threshold through final calibration; no external threshold tuning.'}
 write('inputs/protocol.json',protocol)
 dev=pd.read_csv(SOURCE/'development.csv').set_index('ID');oof=pd.read_csv(SOURCE/'oof_predictions.csv');locked=json.loads((SOURCE/'locked_final.json').read_text());folds=json.loads((SOURCE/'folds.json').read_text());rows=[];calibrators=[];innerrows=[];boundaries=[]
 for fold in range(1,6):
  d=SOURCE/f'fold_{fold}';split=folds[fold-1];tr=set(split['training_ids']);te=set(split['heldout_ids']);assert tr.isdisjoint(te)
  tuning=json.loads((d/'tuning.json').read_text());parts=[]
  for inner in range(1,6):
   sub=d/f'inner_{inner}';s=json.loads((sub/'split.json').read_text());itr=set(s['outer_train']);iva=set(s['outer_test']);assert itr.isdisjoint(iva) and (itr|iva)==tr and (itr|iva).isdisjoint(te)
   x,y,z,ids=matrices(sub,True);a=pd.DataFrame({'ID':ids,'event':dev.loc[ids,'event'].to_numpy(),'inner_fold':inner,'outer_fold':fold})
   for name,family in MODELS.items():
    params=tuning['best'][name]['parameters'] if family=='CatBoost' else {}
    model=estimator(family,params,seed=42);model.fit(x,y);a[name]=model.predict_proba(z)[:,1]
   parts.append(a);boundaries.append({'outer_fold':fold,'inner_fold':inner,'train_n':len(itr),'validation_n':len(iva),'outer_test_overlap':0})
  a=pd.concat(parts);assert a.ID.is_unique and set(a.ID)==tr;innerrows.append(a)
  outer=oof[oof.fold==fold].copy();assert set(outer.ID)==te
  for name in MODELS:
   c=fit(a.event,a[name]);calibrators.append({'stage':f'outer_{fold}','model':name,'fit_n':len(a),**c})
   outer[name+'_calibrated']=apply(outer[name],c)
   assert np.isclose(roc_auc_score(outer.event,outer[name]),roc_auc_score(outer.event,outer[name+'_calibrated']))
  rows.append(outer);print('outer fold',fold,'calibrator complete',flush=True)
 calibrated_oof=pd.concat(rows);calibrated_oof.to_csv(ROOT/'results/development_predictions.csv',index=False);pd.concat(innerrows).to_csv(ROOT/'audit/calibration_training_inner_predictions.csv',index=False)
 final={}
 for name in MODELS:
  c=fit(oof.event,oof[name]);c['raw_threshold']=locked['thresholds'][name];c['calibrated_threshold']=float(apply(c['raw_threshold'],c));final[name]=c;calibrators.append({'stage':'final','model':name,'fit_n':len(oof),**c})
 write('inputs/locked_calibrators.json',final) # Before reading external outcomes.
 ext=pd.read_csv(SOURCE/'final/predictions.csv');labels=pd.read_csv(SOURCE/'evaluation_labels.csv').query("cohort=='external_test'")[['ID','event']];ext=ext.merge(labels,on='ID',validate='one_to_one');assert len(ext)==309
 for name in MODELS:
  ext[name+'_calibrated']=apply(ext[name],final[name]);assert np.array_equal(ext[name]>=final[name]['raw_threshold'],ext[name+'_calibrated']>=final[name]['calibrated_threshold']);assert np.isclose(roc_auc_score(ext.event,ext[name]),roc_auc_score(ext.event,ext[name+'_calibrated']))
 ext.to_csv(ROOT/'results/external_predictions.csv',index=False);pd.DataFrame(calibrators).to_csv(ROOT/'results/calibrators.csv',index=False);pd.DataFrame(boundaries).to_csv(ROOT/'audit/partition_checks.csv',index=False)
 metrics=[];pairs=[]
 for cohort,data in [('development',calibrated_oof),('external',ext)]:
  y=data.event.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
  for name in MODELS:
   for state in ['raw','calibrated']:
    col=name if state=='raw' else name+'_calibrated';p=data[col].to_numpy();cut=final[name]['raw_threshold' if state=='raw' else 'calibrated_threshold'] if cohort=='external' else None
    m=metrics_ci(y,p,cut,rounds=2000);m['Log_loss']=log_loss(y,p,labels=[0,1]);metrics.append({'cohort':cohort,'model':name,'state':state,**m})
   a=data[name].to_numpy();b=data[name+'_calibrated'].to_numpy();a=np.clip(a,1e-6,1-1e-6);b=np.clip(b,1e-6,1-1e-6)
   differences={'Brier':(b-y)**2-(a-y)**2,'Log_loss':-(y*np.log(b)+(1-y)*np.log1p(-b))+(y*np.log(a)+(1-y)*np.log1p(-a))}
   for metric,values in differences.items():
    rng=np.random.default_rng(42);boot=[values[np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]].mean() for _ in range(2000)];lo,hi=np.quantile(boot,[.025,.975]);pairs.append({'cohort':cohort,'model':name,'metric':metric,'calibrated_minus_raw':values.mean(),'CI_low':lo,'CI_high':hi,'interpretation':'Negative favors calibration; fixed predictions bootstrap does not refit training pipeline'})
   print(cohort,name,'metrics complete',flush=True)
 pd.DataFrame(metrics).to_csv(ROOT/'results/metrics.csv',index=False);pd.DataFrame(pairs).to_csv(ROOT/'results/paired_loss_changes.csv',index=False)
 paths=[SOURCE/'oof_predictions.csv',SOURCE/'final/predictions.csv',SOURCE/'locked_final.json',SOURCE/'folds.json']
 write('audit/source_hashes.json',[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
 write('audit/complete.json',{'complete':True,'outer_folds':5,'inner_boundaries_verified':25,'development_n':len(oof),'external_n':len(ext),'external_auc_unchanged':True,'mapped_threshold_decisions_unchanged':True})
if __name__=='__main__':main()
