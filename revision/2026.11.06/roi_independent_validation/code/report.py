from pathlib import Path
import os,sys,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/jmri_mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
# Parallel workers publish fully completed folds before the original coordinator proceeds.
if (ROOT/'plans/parallel_execution.json').exists() and (ROOT/'runs/oof_roi_fold2/heldout_predictions.csv').exists():
 import time
 waiting=ROOT/'audit/coordinator_waiting_for_parallel.json'
 while not all((ROOT/f'audit/parallel_fold{f}_complete.json').exists() for f in [3,4,5]):
  state={}
  for f in [3,4,5]:
   p=ROOT/f'parallel/fold{f}/status.json'
   state[str(f)]=json.loads(p.read_text()) if p.exists() else {'stage':'starting'}
  waiting.write_text(json.dumps(state,ensure_ascii=False,indent=2))
  if any(v.get('stage')=='failed' for v in state.values()):raise RuntimeError('Parallel fold failed; see parallel/fold*/status.json; checkpoints preserved')
  time.sleep(20)
 if waiting.exists():waiting.unlink()
sys.path.insert(0,str(CORE/'reviewer_statistics/code'));from manuscript_tables import Tables
sys.path.insert(0,str(CORE/'code'));from revision_statistics import metrics_ci,threshold,paired
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf');plt.rcParams.update({'font.family':'Times New Roman','pdf.fonttype':42,'font.size':11})
rows=[];histories=[];oof=[]
for f in range(1,6):
 p=ROOT/f'runs/inner_roi_fold{f}/history.csv'
 if not p.exists():continue
 h=pd.read_csv(p);h['fold']=f;histories.append(h)
 done=ROOT/f'runs/inner_roi_fold{f}/complete.json'
 best_epoch=int(json.loads(done.read_text())['best']['60']['epoch']) if done.exists() else int(h.loc[h.validation_auc.idxmax(),'epoch'])
 b=h[h.epoch==best_epoch].iloc[0]
 rows.append(dict(fold=f,epochs_completed=int(h.epoch.max()),best_epoch=best_epoch,validation_auc=float(b.validation_auc),validation_loss=float(b.validation_loss),training_complete=done.exists(),early_stopping=False))
 fig,axes=plt.subplots(1,3,figsize=(13,3.5))
 for ax,key,title in zip(axes,['loss','auc','train_augmented_loss'],['Loss (no augmentation)','AUC','Augmented training loss']):
  if key=='train_augmented_loss':ax.plot(h.epoch,h[key],label='Train')
  else:
   for part in ['train','validation']:ax.plot(h.epoch,h[part+'_'+key],label=part.capitalize())
  ax.set(xlabel='Epoch',ylabel=title,title=f'ROI fold {f}');ax.legend()
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(ROOT/f'figures/ROI_fold{f}_curves.{ext}',dpi=300)
 plt.close(fig)
 p=ROOT/f'runs/oof_roi_fold{f}/heldout_predictions.csv'
 if p.exists():g=pd.read_csv(p);g['fold']=f;oof.append(g)
if histories:pd.concat(histories).to_csv(ROOT/'results/epoch_history.csv',index=False)
t=Tables();t.add('Table ROI1','Independent ROI inner-validation training',['Fold','Epochs completed','Best epoch','Validation AUC','Validation loss','Status'],[[r['fold'],r['epochs_completed'],r['best_epoch'],f"{r['validation_auc']:.4f}",f"{r['validation_loss']:.4f}",'Complete' if r['training_complete'] else 'Provisional'] for r in rows], '60 scheduled epochs; no early stopping. Best epoch selected only using the inner validation subset of the outer training fold. Incomplete-fold maxima are provisional. Development n=1050; existing ROI/full-image fallback rules retained.')
metrics=[];comparisons=[]
if len(oof)==5:
 g=pd.concat(oof);assert len(g)==1050 and g.ID.nunique()==1050;g.to_csv(ROOT/'results/oof_predictions.csv',index=False)
 cut=threshold(g.event,g.probability)
 metrics.append(dict(cohort='development_OOF',**metrics_ci(g.event,g.probability,cut,rounds=1000)))
 p=ROOT/'runs/final_roi/external_predictions.csv'
 if p.exists():
  ext=pd.read_csv(p);assert len(ext)==309;metrics.append(dict(cohort='external_reused',**metrics_ci(ext.event,ext.probability,cut,rounds=1000)))
  old=pd.read_csv(CORE/'runs/final_roi_matched_full/external_predictions.csv');m=old.merge(ext,on='ID',suffixes=('_old','_new'),validate='one_to_one');assert (m.event_old==m.event_new).all();comparisons.append(dict(cohort='external_reused',**paired(m.event_old,m.probability_old,m.probability_new)))
 old=pd.concat([pd.read_csv(CORE/f'runs/oof_roi_matched_full_fold{f}/heldout_predictions.csv') for f in range(1,6)])
 m=old.merge(g,on='ID',suffixes=('_old','_new'),validate='one_to_one');assert (m.event_old==m.event_new).all();comparisons.append(dict(cohort='development_OOF',**paired(m.event_old,m.probability_old,m.probability_new)))
 pd.DataFrame(metrics).to_csv(ROOT/'results/performance.csv',index=False);pd.DataFrame(comparisons).to_csv(ROOT/'results/paired_vs_original_ROI.csv',index=False)
 t.add('Table ROI2','ROI with independent epoch selection',['Cohort','n','AUC [95% CI]','Sensitivity','Specificity','NPV'],[[r['cohort'],r['N'],f"{r['AUC']:.4f} [{r['AUC_CI_low']:.4f}, {r['AUC_CI_high']:.4f}]",f"{r['Sensitivity']:.4f}",f"{r['Specificity']:.4f}",f"{r['NPV']:.4f}"] for r in metrics], 'Threshold selected from development OOF sensitivity >=0.95 and locked before external use; 1000 bootstrap replicates. These are 1050-case ROI results, not the new 1059-case complete pipeline.')
t.doc.save(ROOT/'results/ROI独立验证训练记录.docx');t.wb.save(ROOT/'results/ROI独立验证训练记录.xlsx')
pd.DataFrame(rows).to_csv(ROOT/'results/fold_selection_status.csv',index=False)
if metrics:
 with pd.ExcelWriter(ROOT/'results/ROI独立验证完整结果.xlsx') as w:
  for name,df in [('Training',pd.DataFrame(rows)),('Epoch_history',pd.concat(histories)),('Performance',pd.DataFrame(metrics)),('Paired_comparisons',pd.DataFrame(comparisons))]:df.to_excel(w,sheet_name=name,index=False)
print('ROI report refreshed; completed inner folds:',sum(r['training_complete'] for r in rows),flush=True)
if (ROOT/'runs/final_roi/external_predictions.csv').exists():
 import subprocess
 subprocess.run([sys.executable,str(ROOT/'code/assemble_delivery.py')],check=True)
