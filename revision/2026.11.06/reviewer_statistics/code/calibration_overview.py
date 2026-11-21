from pathlib import Path
import sys,json
import pandas as pd,numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font,Alignment,Border,Side
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(ROOT/'code'))
from manuscript_tables import Tables,ci,num
names={'CMR':'CMR','Rscore_full':'Full Rscore','Rscore_texture':'Non-shape Rscore','CMR_Rscore_full':'CMR + Rscore / CatBoost','CMR_Rscore_texture':'CMR + non-shape Rscore / CatBoost','CMR_Rscore_EN':'CMR + Rscore / elastic net','Selected':'Inner-selected strategy','Rscore':'Rscore','Rscore_CMR':'CMR + Rscore','mViT':'mViT-FT features','mViT_CMR':'CMR + mViT-FT','mViT_Rscore_CMR':'CMR + Rscore + mViT-FT','Frozen':'mViT-Frozen features','Frozen_CMR':'CMR + mViT-Frozen','Frozen_Rscore_CMR':'CMR + Rscore + mViT-Frozen','baseline20_full':'mViT end-to-end / 20-epoch window','extended60_full':'mViT end-to-end / 60-epoch window','roi_matched_full':'mViT ROI / 60-epoch window','Rscore_CMR_CatBoost':'CMR + Rscore / CatBoost','Rscore_CMR_EN':'CMR + Rscore / elastic net'}
def main():
 a=pd.read_csv(ROOT/'tables/performance.csv');a['analysis']='A: Full tabular nested validation';a['development_N']=1059;a['source']='reviewer_statistics/tables/performance.csv'
 b=pd.read_excel(CORE/'nested_validation/tables/nested_validation_results.xlsx',sheet_name='Metrics');b.cohort=b.cohort.map({'development_OOF':'development','external_reused':'external_test'});b['analysis']='B: Matched nine-model comparison';b['development_N']=1050;b['source']='nested_validation/tables/nested_validation_results.xlsx:Metrics'
 old=pd.read_csv(CORE/'tables/Performance.csv').rename(columns={'Model':'model','Dataset':'cohort'});old.cohort=old.cohort.map({'Development OOF':'development','External test':'external_test'});old['source']='tables/Performance.csv'
 c=old[old.model.isin(['baseline20_full','extended60_full','roi_matched_full'])].copy();c['analysis']='C: End-to-end mViT ablations';c['development_N']=1050
 historical=old[~old.model.isin(c.model.unique())].copy();historical['analysis']='Historical / exploratory appendix'
 d=pd.read_csv(CORE/'recalibration/results/metrics.csv');d=d[d.state=='calibrated'].copy();d.cohort=d.cohort.replace({'external':'external_test'});d['analysis']='D: Logistic recalibration sensitivity';d['development_N']=1059;d['source']='recalibration/results/metrics.csv'
 full=pd.concat([a,b,c,d],ignore_index=True);full['Model_label']=full.model.map(names);assert full.Model_label.notna().all()
 full.to_csv(ROOT/'tables/主要模型校准汇总.csv',index=False)
 notes=pd.DataFrame([{'Group':'A','Meaning':'1059 development / 309 external; full nested upstream selection and downstream classifier tuning. Rscore alone uses original fitted probability. Selected and CatBoost share final external probabilities; they are not independent models.'},{'Group':'B','Meaning':'1050 matched development / same 309 external patients; nine models use common outer folds and fixed downstream elastic-net classifier. mViT denotes features plus downstream classifier, not the end-to-end task head.'},{'Group':'C','Meaning':'1050 development / 309 external; original end-to-end mViT probability head, different from group B. Windows define internal epoch selection, not all refits training that many epochs.'},{'Group':'D','Meaning':'Development-fitted logistic recalibration; outer-fold development evaluation; previously evaluated external cohort reused. Separate sensitivity analysis, not a replacement selected on external results.'},{'Group':'All','Meaning':'Intercept and slope jointly fitted; positive intercept is not offset-only calibration-in-the-large. Ideal pair: intercept 0, slope 1. CIs covering ideal values do not prove good calibration. Brier combines discrimination/calibration and depends on outcome prevalence. Historical appendix is not pooled with new comparisons.'}])
 with pd.ExcelWriter(ROOT/'tables/主要模型校准汇总.xlsx') as writer:
  full[full.cohort=='external_test'].to_excel(writer,sheet_name='External_main',index=False)
  full[full.cohort=='development'].to_excel(writer,sheet_name='Development_OOF',index=False)
  historical.to_excel(writer,sheet_name='Historical_exploratory',index=False)
  notes.to_excel(writer,sheet_name='Comparison_notes',index=False)
 wb=load_workbook(ROOT/'tables/主要模型校准汇总.xlsx')
 for ws in wb:
  ws.sheet_view.showGridLines=False;ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions;last=ws.max_row
  for row in ws:
   for cell in row:cell.font=Font(name='Times New Roman',size=12,bold=cell.row==1);cell.alignment=Alignment(wrap_text=True,vertical='top');cell.border=Border(top=Side(style='thin') if cell.row==1 else Side(),bottom=Side(style='thin') if cell.row in [1,last] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=25
 wb.save(ROOT/'tables/主要模型校准汇总.xlsx')
 t=Tables();i=0
 for cohort in ['external_test','development']:
  for group,sub in full[full.cohort==cohort].groupby('analysis',sort=False):
   i+=1
   t.add(f'Table K{i}',f'{cohort}: {group}',['Model','n / LGE+','AUC','Intercept [95% CI]','Slope [95% CI]','Brier'],[[r.Model_label,f'{int(r.N)} / {int(r.Events)}',num(r.AUC),ci(r,'Calibration_intercept_joint'),ci(r,'Calibration_slope_joint'),f'{r.Brier:.4f}'] for _,r in sub.iterrows()],notes.loc[notes.Group==group[0],'Meaning'].iloc[0]+' Intercept and slope are jointly estimated; ideal values are 0 and 1. Brackets denote 95% CIs. External results refer to the same previously evaluated cohort.',widths=[7.1,2.4,2,5.7,5.7,2.2])
 t.doc.save(ROOT/'tables/主要模型校准汇总_稿件样式.docx')
 summary='2026-11-10 主要模型校准汇总\n已将全流程表格模型、共同划分九模型、mViT训练/ROI消融、重新校准敏感性分析分组列出。\n文件：reviewer_statistics/tables/主要模型校准汇总.xlsx；主要模型校准汇总_稿件样式.docx。\n核心发现：单独Rscore的斜率接近1，但联合模型AUC及Brier更好，不能单凭斜率排名。共同划分下，加入微调mViT未改善外部AUC/Brier，斜率也更低。端到端mViT消融均存在外部斜率偏低，60轮或ROI未改善该指标。\n'
 (ROOT/'response/主要模型校准汇总说明.txt').write_text(summary)
 print('Wrote',len(full),'main rows,',len(historical),'historical/exploratory rows,',i,'Word tables.')
if __name__=='__main__':main()
