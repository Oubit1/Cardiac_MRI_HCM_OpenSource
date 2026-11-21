from pathlib import Path
import sys,json
import pandas as pd
R=Path(__file__).resolve().parents[1];C=R.parent
sys.path.insert(0,str(C/'code'))
from revision_statistics import metrics_ci
from sklearn.metrics import roc_curve
import numpy as np
rows=[]
def threshold(y,p):
 f,t,c=roc_curve(y,p);ix=np.where((t>=.95)&np.isfinite(c))[0];return float(c[ix[np.argmax((1-f)[ix])]])
oof=pd.read_csv(R/'results/nine_model_oof_predictions.csv');ext=pd.read_csv(R/'results/nine_model_external_predictions.csv')
models=[x for x in oof if x not in ['ID','event']]
for m in models:
 cut=threshold(oof.event,oof[m])
 for co,d in [('development_OOF',oof),('external',ext)]:rows.append({'model':m,'cohort':co,**metrics_ci(d.event,d[m],cut=cut,rounds=1000)})
a=pd.read_csv(R/'results/roi_oof_predictions.csv');b=pd.read_csv(R/'runs/final_roi/external_predictions.csv');cut=threshold(a.event,a.probability)
for co,d in [('development_OOF',a),('external',b)]:rows.append({'model':'ROI_mViT','cohort':co,**metrics_ci(d.event,d.probability,cut=cut,rounds=1000)})
x=pd.DataFrame(rows);x.to_csv(R/'results/performance_1059.csv',index=False)
with pd.ExcelWriter(R/'results/1059例九模型及ROI完整结果.xlsx') as w:
 x.to_excel(w,sheet_name='性能',index=False);oof.to_excel(w,sheet_name='九模型OOF',index=False);ext.to_excel(w,sheet_name='九模型外部',index=False);a.to_excel(w,sheet_name='ROI_OOF',index=False);b.to_excel(w,sheet_name='ROI外部',index=False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
figdir=R/'figures';figdir.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,data,title in zip(axes,[oof,ext],['Development OOF (n=1059)','External (n=309)']):
 for m in models:
  fpr,tpr,_=roc_curve(data.event,data[m]);ax.plot(fpr,tpr,label=f'{m}: {roc_auc_score(data.event,data[m]):.4f}')
 ax.plot([0,1],[0,1],'k--');ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=title);ax.legend(fontsize=7)
fig.tight_layout()
for suffix in ['png','pdf','tiff']:fig.savefig(figdir/f'Nine_model_ROC_1059.{suffix}',dpi=300)
plt.close(fig)
for mode in ['full','roi']:
 for fold in range(1,6):
  h=pd.read_csv(R/f'runs/inner_{mode}_fold{fold}/history.csv');fig,axs=plt.subplots(1,2,figsize=(10,3.5))
  axs[0].plot(h.epoch,h.train_loss,label='Training');axs[0].plot(h.epoch,h.validation_loss,label='Validation');axs[0].set(xlabel='Epoch',ylabel='Loss')
  axs[1].plot(h.epoch,h.train_auc,label='Training');axs[1].plot(h.epoch,h.validation_auc,label='Validation');axs[1].set(xlabel='Epoch',ylabel='AUC')
  for ax in axs:ax.legend()
  fig.suptitle(f'{mode} fold {fold}, development n=1059');fig.tight_layout()
  for suffix in ['png','pdf']:fig.savefig(figdir/f'{mode}_fold{fold}_curves.{suffix}',dpi=300)
  plt.close(fig)
import subprocess
subprocess.run([sys.executable,'-u',str(R/'code/roi_fusion1059.py')],check=True)
if (C/'Table 2.docx').exists():
 subprocess.run([sys.executable,'-u',str(R/'code/fill_table2.py')],check=True)
(R/'COMPLETE.json').write_text(json.dumps({'complete':True,'development_n':1059,'external_n':309,'models':models+['ROI_mViT'],'note':'new run; historical reports not silently replaced'},indent=2))
