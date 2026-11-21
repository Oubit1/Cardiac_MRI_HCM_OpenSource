"""Assemble current manuscript deliverables; label pending ROI explicitly until complete."""
from pathlib import Path
import sys,json,shutil
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.calibration import calibration_curve
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent;OUT=CORE/'delivery/JMRI交付包'
for name in ['文档','图片','数据','代码']:(OUT/name).mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf');plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 fig.tight_layout()
 for ext in ['pdf','svg','png','tiff']:fig.savefig(OUT/f'图片/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
D=pd.read_csv(CORE/'reviewer_statistics/tables/predictions_development.csv');E=pd.read_csv(CORE/'reviewer_statistics/tables/predictions_external_test.csv');model='CMR_Rscore_full'
sys.path.insert(0,str(CORE/'code'));from revision_statistics import threshold,estimates
fig,axes=plt.subplots(2,2,figsize=(10,8))
cut=threshold(D.event,D[model])
for j,(g,label) in enumerate([(D,'Development OOF (n=1059)'),(E,'External (n=309)')]):
 ax=axes[0,j]
 for key,title in [('CMR','Without Rscore'),(model,'With Rscore')]:
  x,y,_=roc_curve(g.event,g[key]);ax.plot(x,y,label=f'{title}: {roc_auc_score(g.event,g[key]):.4f}')
 ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=f'{chr(65+j)}. {label}');ax.legend(fontsize=9)
 v=estimates(g.event,g[model],cut);matrix=np.array([[v['TN'],v['FP']],[v['FN'],v['TP']]])
 ax=axes[1,j];ax.imshow(matrix,cmap='Blues');ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['LGE-negative','LGE-positive'],yticklabels=['LGE-negative','LGE-positive'],xlabel='Predicted class',ylabel='True class',title=f'{chr(67+j)}. {label}')
 for y in range(2):
  for x in range(2):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',color='white' if matrix[y,x]>matrix.max()/2 else 'black',fontsize=15)
save(fig,'Figure2_CMR_Rscore_ROC')
fig,axes=plt.subplots(1,2,figsize=(11,5),subplot_kw={'projection':'polar'})
keys=['AUC','Accuracy','Sensitivity','Specificity','PPV','NPV','F1'];theta=np.linspace(0,2*np.pi,len(keys),endpoint=False);theta=np.r_[theta,theta[0]]
radar=[]
for ax,g,label in zip(axes,[D,E],['A. Development OOF','B. External']):
 for key,title in [('CMR','Without Rscore'),(model,'With Rscore')]:
  v=estimates(g.event,g[key],threshold(D.event,D[key]));v['F1']=2*v['TP']/(2*v['TP']+v['FP']+v['FN']);values=[v[k] for k in keys];ax.plot(theta,values+[values[0]],label=title);ax.fill(theta,values+[values[0]],alpha=.08);radar.append({'cohort':label,'model':key,**v})
 ax.set_xticks(theta[:-1]);ax.set_xticklabels(keys);ax.set_ylim(0,1);ax.set_title(label,pad=25);ax.legend(loc='lower center',bbox_to_anchor=(.5,-.3))
save(fig,'Figure4_Classification_profiles')
pd.DataFrame(radar).to_csv(OUT/'数据/Figure4_radar_values.csv',index=False)
D.to_csv(OUT/'数据/主图开发集预测.csv',index=False);E.to_csv(OUT/'数据/主图外部预测.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,g,label in zip(axes,[D,E],['A. Development OOF','B. External']):
 obs,pred=calibration_curve(g.event,g[model],n_bins=10,strategy='quantile');ax.plot(pred,obs,'o-',label='CatBoost');ax.plot([0,1],[0,1],'--',color='gray',label='Ideal');ax.set(xlabel='Mean predicted probability',ylabel='Observed proportion',title=label,xlim=(0,1),ylim=(0,1));ax.legend()
save(fig,'SupplementaryFigureS1_Calibration')
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,g,label in zip(axes,[D,E],['A. Development OOF','B. External']):
 ts=np.linspace(.01,.95,95);prev=g.event.mean();ax.plot(ts,prev-(1-prev)*ts/(1-ts),'--',label='Treat all');ax.axhline(0,c='gray',label='Treat none')
 for key,title in [('CMR','CMR'),(model,'CMR + Rscore')]:
  vals=[]
  for t in ts:
   pos=g[key]>=t;vals.append(((pos&(g.event==1)).sum()-(pos&(g.event==0)).sum()*t/(1-t))/len(g))
  ax.plot(ts,vals,label=title)
 ax.set(xlabel='Threshold probability',ylabel='Net benefit',title=label,ylim=(-.1,.8));ax.legend()
save(fig,'SupplementaryFigureS3_DCA')
# Full-transfer curves from completed internal selection; ROI curves from ongoing independent selection.
for branch,paths in [('FT',[CORE/f'runs/inner_full_fold{f}_lr1e-05/history.csv' for f in range(1,6)]),('ROI',[ROOT/f'runs/inner_roi_fold{f}/history.csv' for f in range(1,6)])]:
 for f,path in enumerate(paths,1):
  if not path.exists():continue
  h=pd.read_csv(path);fig,axes=plt.subplots(1,2,figsize=(10,3.8))
  for ax,key in zip(axes,['loss','auc']):
   for part in ['train','validation']:ax.plot(h.epoch,h[part+'_'+key],label=part.capitalize())
   ax.set(xlabel='Epoch',ylabel=key.capitalize(),title=f'{branch} fold {f}');ax.legend()
  save(fig,f'{branch}_Fold{f}_Training_validation')
  shutil.copy2(path,OUT/f'数据/{branch}_Fold{f}_history.csv')
ready=(ROOT/'runs/final_roi/external_predictions.csv').exists() and (ROOT/'results/oof_predictions.csv').exists()
if ready:
 x=CORE/'nested_validation/tables/nested_validation_results.xlsx';fd=pd.read_excel(x,'development_OOF');fe=pd.read_excel(x,'external_reused')
 td=pd.concat([pd.read_csv(CORE/f'runs/oof_extended60_full_fold{f}/heldout_predictions.csv') for f in range(1,6)]);te=pd.read_csv(CORE/'runs/final_extended60_full/external_predictions.csv');rd=pd.read_csv(ROOT/'results/oof_predictions.csv');re=pd.read_csv(ROOT/'runs/final_roi/external_predictions.csv')
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,sets,label in zip(axes,[[(fd,'Frozen','Frozen + classifier'),(td,'probability','FT MIL'),(rd,'probability','ROI MIL')],[(fe,'Frozen','Frozen + classifier'),(te,'probability','FT MIL'),(re,'probability','ROI MIL')]],['A. Development OOF (n=1050)','B. External (n=309)']):
  for g,key,title in sets:
   fpr,tpr,_=roc_curve(g.event,g[key]);ax.plot(fpr,tpr,label=f'{title}: {roc_auc_score(g.event,g[key]):.4f}')
  ax.plot([0,1],[0,1],'--',c='gray');ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=label);ax.legend(fontsize=9)
 save(fig,'mViT_Frozen_FT_ROI_ROC')
if ready:
 import subprocess
 from openpyxl import load_workbook
 delivery=CORE/'delivery/JMRI_filled_2026-11-10'
 updates={};status=pd.read_csv(ROOT/'results/fold_selection_status.csv');metrics=pd.read_csv(ROOT/'results/performance.csv')
 assert len(status)==5 and status.training_complete.all()
 for r in status.itertuples():
  for col,value in [(2,str(r.best_epoch)),(3,f'{r.validation_auc:.4f}'),(4,f'{r.validation_loss:.4f}'),(5,'否；固定60轮内独立选优')]:updates[f'5,{int(r.fold)+5},{col}']=value
 for cohort,col in [('development_OOF',1),('external_reused',2)]:
  r=metrics[metrics.cohort==cohort].iloc[0];updates[f'4,3,{col}']=f'{r.AUC:.4f} [{r.AUC_CI_low:.4f}, {r.AUC_CI_high:.4f}]'
 (delivery/'audit/roi_table_updates.json').write_text(json.dumps(updates,ensure_ascii=False,indent=2))
 subprocess.run([sys.executable,str(delivery/'restore_original_format.py')],check=True)
 wp=delivery/'JMRI_配套完整数据.xlsx';wb=load_workbook(wp)
 for key,value in updates.items():
  ti,ri,ci=map(int,key.split(','));wb[f'Table_{ti+1:02}'].cell(ri+1,ci+1,value)
 wb.save(wp)
 subprocess.run([sys.executable,str(delivery/'sync_display.py')],check=True)
 shutil.copy2(ROOT/'results/ROI独立验证完整结果.xlsx',OUT/'数据/ROI独立验证完整结果.xlsx')
 (OUT/'文档/ROI训练表更新说明.txt').write_text('ROI五折独立内部验证及外层重拟合已完成；主文档三分支表ROI行及训练记录ROI五行已更新。早期缺项清单中“ROI独立验证曲线未完成”状态由本说明更新。结果仍为1050例开发/309例外部。')
for name in ['JMRI_已填结果_保留原格式.docx','缺项与完成状态.txt','区间展示统一说明.txt','JMRI_配套完整数据.xlsx']:
 p=CORE/'delivery/JMRI_filled_2026-11-10'/name
 if p.exists():shutil.copy2(p,OUT/('数据' if p.suffix=='.xlsx' else '文档')/name)
for name in ['report.py','assemble_delivery.py','run.py']:shutil.copy2(ROOT/'code'/name,OUT/'代码'/name)
notes=['交付目录持续更新，训练尚未全部结束时不是最终提交包。','文档保留用户原始JMRI模板格式。','Figure2/4、S1/S3基于1059例表格模型；mViT比较仍为1050例。','FT为原完整60轮内部训练；ROI仅在新训练完成后形成完整曲线和ROC。','所有新绘图采用Times New Roman，PDF/SVG可编辑矢量图，PNG/TIFF 300 dpi。','Frozen采用冻结编码器+分类器，FT/ROI采用MIL读出，图例明确区分。','S1折线为分箱观察比例，不是校准斜率；截距/斜率见统计表。','S2 SHAP、Figure1流程图、Figure3代表病例仍需与最终稿统一，暂不将旧版图标为更新完成。','绘图脚本依赖当前项目源文件，源数据与训练日志见数据目录及项目原路径。']
(OUT/'交付说明.txt').write_text('\n'.join(notes));(OUT/'交付状态.json').write_text(json.dumps({'roi_complete':ready,'final_submission_ready':False,'pending':['SHAP S2 update','Figure1 flow and Figure3 case review','1059-case unified benchmark and outstanding clinical evidence']},ensure_ascii=False,indent=2))
print('Delivery updated. ROI final available:',ready,flush=True)
