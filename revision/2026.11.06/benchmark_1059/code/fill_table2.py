from pathlib import Path
import sys,json,zipfile,copy,os
import numpy as np,pandas as pd
from sklearn.metrics import roc_curve
from lxml import etree as E
R=Path(__file__).resolve().parents[1];C=R.parent;sys.path.insert(0,str(C/'code'))
from revision_statistics import metrics_ci
P=C/'Table 2.docx';backup=C/'archive/Table2_original_20261115.docx';backup.parent.mkdir(exist_ok=True)
if not backup.exists():backup.write_bytes(P.read_bytes())
maprows={2:'CMR',3:'Rscore',4:'Frozen',5:'mViT',7:'Rscore_CMR',8:'Frozen_CMR',9:'mViT_CMR',11:'Frozen_Rscore_CMR',13:'mViT_Rscore_CMR'}
rows=[]
def calculate(prefix,models):
 a=pd.read_csv(R/f'results/{prefix}_oof_predictions.csv');b=pd.read_csv(R/f'results/{prefix}_external_predictions.csv');assert len(a)==1059 and len(b)==309
 cache=R/f'results/{prefix}_table2_metrics.csv'
 if cache.exists():return pd.read_csv(cache)
 rr=[]
 for m in models:
  fp,tp,thresholds=roc_curve(a.event,a[m]);ok=np.where((tp>=.95)&np.isfinite(thresholds))[0];cut=float(thresholds[ok[np.argmax((1-fp)[ok])]])
  for cohort,d in [('development',a),('external',b)]:
   rr.append({'model':m,'cohort':cohort,**metrics_ci(d.event,d[m],cut=cut,rounds=2000)})
 out=pd.DataFrame(rr);out.to_csv(cache,index=False);return out
x=calculate('nine_model',list(maprows.values()))
roi=(R/'results/roi_fusion_complete.json').exists()
if roi:
 y=calculate('roi_fusion',['mViT','mViT_CMR','mViT_Rscore_CMR'])
else:y=None
with zipfile.ZipFile(P) as z:entries=[(i,z.read(i.filename)) for i in z.infolist()]
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};w='{'+ns['w']+'}'
xm=E.fromstring(dict((i.filename,b) for i,b in entries)['word/document.xml']);trs=xm.find('.//w:tbl',ns).findall('w:tr',ns)
def setcell(tc,value):
 ts=tc.findall('.//w:t',ns)
 if ts:
  ts[0].text=value
  for t in ts[1:]:t.text=''
 else:
  p=tc.find('w:p',ns)
  if p is None:p=E.SubElement(tc,w+'p')
  r=p.find('w:r',ns)
  if r is None:
   r=E.SubElement(p,w+'r');sample=trs[2].find('w:tc/w:p/w:r/w:rPr',ns)
   if sample is not None:r.append(copy.deepcopy(sample))
  E.SubElement(r,w+'t').text=value

def fill(i,m,data):
 a=data[(data.model==m)&(data.cohort=='development')].iloc[0];b=data[(data.model==m)&(data.cohort=='external')].iloc[0]
 def auc(t):return f'{t.AUC:.4f} [{t.AUC_CI_low:.4f}, {t.AUC_CI_high:.4f}]'
 vals=[auc(a),auc(b)]+[f'{b[k]:.4f}' for k in ['Sensitivity','Specificity','NPV','Brier']]
 for tc,v in zip(trs[i].findall('w:tc',ns)[1:],vals):setcell(tc,v)
for i,m in maprows.items():fill(i,m,x)
for i,m in {6:'mViT',10:'mViT_CMR',12:'mViT_Rscore_CMR'}.items():
 if roi:
  fill(i,m,y)
  if i==6:setcell(trs[i].findall('w:tc',ns)[0],'mViT-ROI')
 else:
  for tc in trs[i].findall('w:tc',ns)[1:]:setcell(tc,'Pending ROI completion')
note='Note: All strategies use 1059 development patients and 309 external patients. Development estimates are outer five-fold OOF predictions. CMR selection and Rscore construction are repeated within the relevant training partitions; supervised mViT training and epoch selection exclude outer-heldout patients. All rows use the same fixed elastic-net logistic downstream classifier. Sensitivity, specificity, NPV and Brier score refer to the external cohort; each threshold is selected from development OOF predictions targeting sensitivity >=0.95 and then fixed for external evaluation. Brackets indicate 95% bootstrap CIs (2000 replicates) for AUC. This is one outer five-fold evaluation, not five repeated runs. ROI indicates myocardium-guided cropping; its single-model row uses the same downstream classifier as the other rows, rather than the standalone MIL head. AUC, area under the receiver operating characteristic curve; CMR, cardiovascular magnetic resonance; FT, fine-tuning; NPV, negative predictive value; OOF, out-of-fold.'
# Merged note continues through rows: modify only text-bearing cells, preserving merges.
for tr in trs[14:]:
 for tc in tr.findall('w:tc',ns):
  if tc.findall('.//w:t',ns):setcell(tc,note)
tmp=P.with_suffix('.tmp.docx')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for i,b in entries:z.writestr(i,E.tostring(xm,xml_declaration=True,encoding='UTF-8',standalone=True) if i.filename=='word/document.xml' else b)
os.replace(tmp,P)
allmetrics=pd.concat([x.assign(branch='full'),y.assign(branch='roi')] if roi else [x.assign(branch='full')]);allmetrics.to_excel(R/'results/Table2_metrics_1059.xlsx',index=False)
(R/'results/Table2_update_status.json').write_text(json.dumps({'filled_non_roi_rows':9,'filled_roi_rows':3 if roi else 0,'path':str(P),'bootstrap':2000},indent=2))
print('Table 2 updated; ROI complete=',roi,flush=True)
