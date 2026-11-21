"""Read-only follow-up: audited ROI fallback, paired strata, actual cached input panels."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from docx import Document
from docx.shared import Inches,Pt
from docx.oxml.ns import qn
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import metrics_ci,paired
sys.path.insert(0,str(ROOT/'code'))
from manuscript_tables import Tables,ci,num,interval
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf','tiff']:fig.savefig(ROOT/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 x=pd.read_csv(CORE/'audit/cache_slice_index.csv');assert len(x)==10855
 p=x.groupby(['cohort','ID']).roi_fallback.agg(fallback_slices='sum',slices='count').reset_index()
 p['fallback_fraction']=p.fallback_slices/p.slices
 p['group']=np.where(p.fallback_slices==0,'No fallback','Any fallback')
 p.to_csv(ROOT/'tables/ROI_patient_audit.csv',index=False)
 summary=[]
 for cohort,g in x.groupby('cohort'):
  patients=p[p.cohort==cohort]
  summary.append(dict(cohort=cohort,patients=len(patients),slices=len(g),fallback_slices=int(g.roi_fallback.sum()),fallback_slice_fraction=g.roi_fallback.mean(),patients_any_fallback=int((patients.fallback_slices>0).sum()),patients_all_fallback=int((patients.fallback_slices==patients.slices).sum()),patients_no_fallback=int((patients.fallback_slices==0).sum())))
 summary=pd.DataFrame(summary);summary.to_csv(ROOT/'tables/ROI_fallback_summary.csv',index=False)
 # Frozen pre-existing thresholds, no tuning within external strata.
 perf=pd.read_csv(CORE/'tables/Performance.csv');results=[];comparisons=[]
 a=pd.read_csv(CORE/'runs/final_extended60_full/external_predictions.csv').rename(columns={'probability':'full'})
 b=pd.read_csv(CORE/'runs/final_roi_matched_full/external_predictions.csv').rename(columns={'probability':'roi'})
 d=a.merge(b,on=['ID','event'],validate='one_to_one').merge(p.query("cohort=='external_test'"),on='ID',validate='one_to_one');assert len(d)==309
 for group,g in [('All external',d)]+list(d.groupby('group',sort=True)):
  for name,key in [('extended60_full','full'),('roi_matched_full','roi')]:
   cut=float(perf[(perf.Model==name)&(perf.Dataset=='External test')].iloc[0].Threshold)
   results.append({'group':group,'model':name,**metrics_ci(g.event,g[key],cut,rounds=2000)})
  comparisons.append({'group':group,'N':len(g),'Events':int(g.event.sum()),**paired(g.event,g.full,g.roi,rounds=2000)})
 results=pd.DataFrame(results);comparisons=pd.DataFrame(comparisons)
 results.to_csv(ROOT/'tables/ROI_fallback_performance.csv',index=False);comparisons.to_csv(ROOT/'tables/ROI_fallback_paired.csv',index=False)
 d.to_csv(ROOT/'audit/ROI_external_patient_predictions.csv',index=False)
 # Image examples selected by cohort/fallback, then lowest ID and central eligible slice; no outcome/probability criterion.
 chosen=[];fig,axes=plt.subplots(4,4,figsize=(11,11))
 for i,(cohort,fallback) in enumerate([('development',False),('external_test',False),('development',True),('external_test',True)]):
  pool=x[(x.cohort==cohort)&(x.roi_fallback==fallback)].sort_values(['ID','slice_index']);pool=pool[pool.ID==pool.ID.iloc[0]];r=pool.iloc[len(pool)//2]
  raw=np.asarray(nib.load(r.current_path).dataobj,dtype=np.float32);mask=np.asarray(nib.load(r.mask_path).dataobj)>0
  assert raw.shape==mask.shape
  assert hashlib.sha256(Path(r.current_path).read_bytes()).hexdigest()==r.image_sha256
  frame=0;lo,hi=r.normalization_low,r.normalization_high;im=(np.clip(raw[frame],lo,hi)-lo)/max(hi-lo,1e-6)
  for ax in axes[i]:ax.axis('off')
  axes[i,0].imshow(im,cmap='gray',vmin=0,vmax=1);axes[i,0].set_title(f'{"Development" if cohort=="development" else "External"}; '+('fallback' if fallback else 'ROI crop'),fontsize=10)
  axes[i,1].imshow(im,cmap='gray',vmin=0,vmax=1)
  axes[i,1].imshow(np.ma.masked_where(~mask[frame],mask[frame]),cmap='autumn',alpha=.45,vmin=0,vmax=1)
  bbox=None if fallback else json.loads(r.roi_bbox)
  if bbox:
   top,left,h,w=bbox;axes[i,1].add_patch(Rectangle((left,top),w,h,fill=False,edgecolor='cyan',linewidth=1))
  axes[i,1].set_title('Existing mask'+(' + crop box' if bbox else ' (QC fallback)'),fontsize=10)
  for j,key in [(2,'full_cache'),(3,'roi_cache')]:
   cached=np.load(r[key],mmap_mode='r');h,w=cached.shape[-2:];view=cached[frame,(h-224)//2:(h-224)//2+224,(w-224)//2:(w-224)//2+224]
   assert view.shape==(224,224)
   axes[i,j].imshow(view,cmap='gray',vmin=0,vmax=1);axes[i,j].set_title('Full input, 224 × 224' if j==2 else 'ROI input, 224 × 224',fontsize=10)
  if fallback:assert r.full_cache==r.roi_cache
  chosen.append(dict(panel_row=i+1,cohort=cohort,ID=int(r.ID),slice_index=int(r.slice_index),frame=frame,fallback=bool(fallback),reason=r.roi_reason if fallback else '',image_sha256=r.image_sha256,mask_sha256=hashlib.sha256(Path(r.mask_path).read_bytes()).hexdigest(),image_path=r.current_path,mask_path=r.mask_path))
 save(fig,'R8_ROI_actual_inputs');pd.DataFrame(chosen).to_csv(ROOT/'audit/ROI_example_manifest.csv',index=False)
 # Read recorded epoch-level trajectories, without training/inference or checkpoint replacement.
 fig,axes=plt.subplots(2,2,figsize=(10,7));hist=[]
 segroot=Path('/data1/wangjiahui/Code/Automated_Annotation_of_Cine/runs')
 for j,run in enumerate(['unet2d_myo_all_centers','reproduce_cmr']):
  h=pd.read_csv(segroot/run/'history.csv');h['run']=run;hist.append(h);best=h.loc[h.val_dice.idxmax(),'epoch']
  axes[0,j].plot(h.epoch,h.train_loss,label='Training loss');axes[0,j].set(title=run,xlabel='Epoch',ylabel='Training loss')
  for col,label in [('val_dice','Overall'),('val_dice_anzhen','Anzhen'),('val_dice_renji','Renji'),('val_dice_shengyi','Shengyi')]:axes[1,j].plot(h.epoch,h[col],label=label)
  axes[1,j].axvline(best,color='gray',ls='--',label=f'Selected epoch {int(best)}');axes[1,j].set(xlabel='Epoch',ylabel='Pooled-pixel validation Dice');axes[1,j].legend(fontsize=8)
 save(fig,'R7_segmentation_learning_curves');pd.concat(hist).to_csv(ROOT/'tables/Segmentation_epoch_history.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,3.8));y=np.arange(len(comparisons));ax.errorbar(comparisons.Delta_AUC_B_minus_A,y,xerr=[comparisons.Delta_AUC_B_minus_A-comparisons.Delta_CI_low,comparisons.Delta_CI_high-comparisons.Delta_AUC_B_minus_A],fmt='o',capsize=4);ax.axvline(0,color='gray',ls='--');ax.set(yticks=y,yticklabels=[f'{r.group} (n={r.N})' for _,r in comparisons.iterrows()],xlabel='AUC difference: ROI − full [95% CI]');save(fig,'R9_ROI_fallback_paired')
 t=Tables()
 t.add('Table R12','ROI fallback at slice and patient levels',['Cohort','Patients','Slices','Fallback slices, n (%)','Patients with any fallback, n (%)','Patients with all slices falling back'],[[r.cohort,int(r.patients),int(r.slices),f'{r.fallback_slices} ({100*r.fallback_slice_fraction:.2f})',f'{r.patients_any_fallback} ({100*r.patients_any_fallback/r.patients:.2f})',int(r.patients_all_fallback)] for _,r in summary.iterrows()], 'All selected slices and all patients were retained. Existing cine masks underwent reference-free preprocessing checks. Failed slices used the full image. No manual-reference segmentation accuracy is implied by passing these checks.')
 t.add('Table R13','Exploratory external ROI comparison by fallback status',['Stratum','n / LGE+','Full AUC [95% CI]','ROI AUC [95% CI]','Paired ΔAUC [95% CI]'],[[r.group,f'{r.N} / {r.Events}',ci(results[(results.group==r.group)&(results.model=='extended60_full')].iloc[0],'AUC'),ci(results[(results.group==r.group)&(results.model=='roi_matched_full')].iloc[0],'AUC'),interval(r,'Delta_AUC_B_minus_A','Delta_CI_low','Delta_CI_high')] for _,r in comparisons.iterrows()], 'Post hoc descriptive strata determined solely by recorded preprocessing fallback, without outcome-based exclusions or subgroup tuning. Differences are ROI minus full-image predictions for the same patients, using 2000 patient-stratified bootstrap samples. Overlapping all-patient and subgroup estimates are not independent. This is not a causal test of the effect of fallback.')
 t.doc.save(ROOT/'tables/ROI追加表_稿件样式.docx');t.wb.save(ROOT/'tables/ROI追加表_稿件样式.xlsx')
 (ROOT/'audit/ROI_followup_table_index.json').write_text(json.dumps(t.index,ensure_ascii=False,indent=2))
 doc=Document();st=doc.styles['Normal'];st.font.name='Times New Roman';st.font.size=Pt(12);st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 doc.add_heading('ROI回退及分割训练证据：追加中文回复',0)
 doc.add_paragraph('针对输入公平性，我们补充了实际预处理图像、心肌标签及模型224×224输入对照，并报告了回退整图的切片和患者数量。示例按队列与回退状态确定后，选择最小ID的中间合格切片，不依据LGE结局或模型性能选择。图片展示的是现有标签，不是新增人工分割参考。')
 for _,r in summary.iterrows():doc.add_paragraph(f'{r.cohort}：{r.patients}例、{r.slices}张切片；{r.fallback_slices}张（{100*r.fallback_slice_fraction:.2f}%）回退整图；{r.patients_any_fallback}例（{100*r.patients_any_fallback/r.patients:.2f}%）至少包含一张回退切片，没有患者全部切片回退。')
 for _,r in comparisons.iterrows():doc.add_paragraph(f'{r.group}，n={r.N}：ROI相对整图的AUC差值为{r.Delta_AUC_B_minus_A:.4f}（95% CI {r.Delta_CI_low:.4f}至{r.Delta_CI_high:.4f}）。')
 doc.add_paragraph('以上分层为补充探索分析，未重新选择外部阈值，也未按结果剔除病例。分层之间病例构成不同，不能将差异解释为回退整图的因果效应；主要ROI比较仍使用完整309例。')
 doc.add_paragraph('分割项目两套正式运行的训练损失、总体及三个中心的验证Dice曲线已整理，分别标明第5轮和第13轮最佳检查点。源记录没有验证损失列，因此未将训练损失或Dice冒充验证损失。三中心106例为选模验证，实际HCM Mask对应哪套权重仍需推理来源确认。')
 for name,caption in [('R7_segmentation_learning_curves','Figure R7. 两次正式分割运行的训练损失与逐中心验证Dice。虚线为已保存最佳检查点的选择轮次。'),('R8_ROI_actual_inputs','Figure R8. 第0帧原始cine、现有Mask及裁剪框、整图输入与ROI输入。蓝框表示时间维Mask并集加边距后的裁剪范围；回退切片两种输入相同。'),('R9_ROI_fallback_paired','Figure R9. 保留所有病例的主比较，以及按回退状态划分的探索性配对AUC差及95%区间。')]:
  doc.add_picture(str(ROOT/f'figures/{name}.png'),width=Inches(6.2));doc.add_paragraph(caption)
 doc.save(ROOT/'response/ROI追加回复_中文版.docx');(ROOT/'response/ROI追加回复_中文版.txt').write_text('\n'.join(p.text for p in doc.paragraphs))
 (ROOT/'audit/ROI_followup_complete.json').write_text(json.dumps({'complete':True,'original_labels_modified':False,'training_started':False,'external_patients':len(d),'example_selection':'cohort/fallback then minimum ID and middle eligible slice, frame 0','main_comparison_retains_all_patients':True},indent=2))
 print(summary.to_string(index=False));print(comparisons[['group','N','Delta_AUC_B_minus_A','Delta_CI_low','Delta_CI_high']].to_string(index=False))
if __name__=='__main__':main()
