from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches,Pt
from docx.oxml.ns import qn
from openpyxl import load_workbook
from openpyxl.styles import Font,Alignment,Border,Side
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(CORE/'reviewer_statistics/code'))
from manuscript_tables import Tables,num,pval
LABEL={'field_strength_t':'Field strength (3 vs 1.5 T)','pixel_mm':'Pixel spacing','slice_thickness_mm':'Slice thickness','spacing_between_slices_mm':'Center-to-center slice spacing','repetition_time_ms':'DICOM-reported TR','echo_time_ms':'TE','flip_angle_deg':'Flip angle','receiver_bandwidth_hz_px':'Receiver bandwidth','reconstructed_fov_geomean_mm':'Reconstructed FOV'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf','tiff']:fig.savefig(ROOT/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 a=pd.read_csv(ROOT/'results/Associations.csv');missing=pd.read_csv(ROOT/'results/Missingness.csv');v=pd.read_csv(ROOT/'results/VIF.csv');s=pd.read_csv(ROOT/'results/CMR_cohort_shift.csv');cmp=pd.read_csv(ROOT/'results/Archived_new_same_label_comparison.csv');sens=pd.read_csv(ROOT/'results/VIF_sensitivity.csv');diag=pd.read_csv(ROOT/'results/Joint_diagnostics.csv')
 t=Tables();idx=0
 for analysis in ['single_factor_adjusted','joint']:
  for cohort in ['development','external_test']:
   idx+=1;sub=a[(a.analysis==analysis)&(a.cohort==cohort)&(a.outcome=='full_score')]
   t.add(f'Table A{idx}',f'Full Rscore acquisition associations: {cohort}, {analysis}',['Parameter','n','Effect [95% CI]','Holm P'],[[LABEL[r.factor],int(r.N),f'{r.coefficient:.3f} [{r.CI_low:.3f}–{r.CI_high:.3f}]',pval(r.Holm_P)] for _,r in sub.iterrows()], 'Effects are in development-OOF Rscore SD units per analysis-set SD of each continuous predictor; field strength compares 3 T with 1.5 T. Models adjust for LGE and, in development, outer fold. Single-factor models additionally adjust field strength when variable; joint models include all retained technical factors. HC3 normal CIs. Associations do not establish causal scanner effects. Center cannot be adjusted; high VIF is documented.')
 for cohort in ['development','external_test']:
  idx+=1;sub=a[(a.analysis=='joint')&(a.cohort==cohort)&(a.outcome=='texture_score')]
  t.add(f'Table A{idx}',f'Non-shape Rscore acquisition associations: {cohort}',['Parameter','n','Effect [95% CI]','Holm P'],[[LABEL[r.factor],int(r.N),f'{r.coefficient:.3f} [{r.CI_low:.3f}–{r.CI_high:.3f}]',pval(r.Holm_P)] for _,r in sub.iterrows()], 'Secondary outcome, scaled by non-shape development OOF mean/SD. Same joint model and HC3 inference as full Rscore; complete cases. Multiple correlated acquisition factors limit attribution to individual parameters.')
 idx+=1;t.add(f'Table A{idx}','Archived and new predictions evaluated on identical current external labels',['Model','n','Archived AUC','New AUC','New − archived [95% CI]','Holm P'],[[r.model,int(r.N),num(r.archived_probability_AUC_current_labels),num(r.new_probability_AUC_current_labels),f'{r.Delta_AUC_B_minus_A:.4f} [{r.Delta_CI_low:.4f}–{r.Delta_CI_high:.4f}]',pval(r.Holm_P)] for _,r in cmp.iterrows()], 'All comparisons use the same 309 patients and current 212 positive labels. Archived probabilities are from local result files; they are not assumed to reproduce every manuscript value. Paired patient bootstrap (2000 replicates) and Holm correction over three DeLong comparisons. Intervals spanning zero do not establish equivalence.')
 idx+=1;t.add(f'Table A{idx}','Observed conventional MRI distribution shift',['Variable','Development mean ± SD','External mean ± SD','SMD (external − development)'],[[r.variable,f'{r.development_mean:.2f} ± {r.development_SD:.2f}',f'{r.external_mean:.2f} ± {r.external_SD:.2f}',num(r.SMD_external_minus_development)] for _,r in s.iterrows() if r.variable!='sex'], 'Development n=1059 and external n=309 before variable-specific missingness. SMD divides the signed mean difference by the square root of the mean within-cohort variances. Descriptive distributions only; no causal attribution to scanner or proof that distribution shift caused calibration bias. Sex coding is retained only in the detailed workbook.')
 idx+=1;rows=[]
 for _,r in missing.iterrows():rows.append([r.cohort,LABEL[r.factor],f'{int(r.valid_n)} / {int(r.metadata_patients)}',f'{r["median"]:.2f} [{r.Q1:.2f}–{r.Q3:.2f}]'])
 # Split this table by cohort to retain readable manuscript typography.
 for cohort in ['development','external_test']:
  sub=[r for r in rows if r[0]==cohort];t.add(f'Table A{idx}',f'Acquisition data availability: {cohort}',['Cohort','Parameter','Available / metadata n','Median [IQR]'],sub,'Metadata cover 1050/1059 development and 309/309 external patients. Receiver bandwidth is read from the same selected cine series. Reconstructed FOV derives from image dimensions and spacing, not necessarily prescribed acquisition FOV.');idx+=1
 t.doc.add_paragraph('Parameter provenance: TR is the DICOM-reported RepetitionTime, not yet reconciled with nominal protocol TR in manuscript Table S1. External median is 43.94 ms versus nominal approximately 3 ms; no conversion has been assumed. Analyses involving this parameter remain exploratory. The external VIF sensitivity model excludes it.')
 t.doc.save(ROOT/'results/补充采集分析与同标签对照_稿件样式.docx');t.wb.save(ROOT/'results/补充采集分析与同标签对照_稿件样式.xlsx')
 wb=load_workbook(ROOT/'results/采集因素与新Rscore_完整结果.xlsx')
 for sheet,file in [('VIF_sensitivity','VIF_sensitivity.csv'),('VIF_removals','VIF_removals.csv'),('VIF_sensitivity_diagnostics','VIF_sensitivity_diagnostics.csv'),('Same_label_comparison','Archived_new_same_label_comparison.csv')]:
  if sheet in wb:del wb[sheet]
  ws=wb.create_sheet(sheet);df=pd.read_csv(ROOT/f'results/{file}').fillna('');ws.append(list(df.columns))
  for row in df.itertuples(index=False,name=None):ws.append(list(row))
 for ws in wb:
  ws.sheet_view.showGridLines=False;ws.freeze_panes='A2';last=ws.max_row
  for row in ws:
   for c in row:c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.alignment=Alignment(wrap_text=True,vertical='top');c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=24
 if 'Parameter_provenance' in wb:del wb['Parameter_provenance']
 ws=wb.create_sheet('Parameter_provenance');ws.append(['Parameter','Qualification']);ws.append(['DICOM-reported TR','External median 43.94 ms; manuscript nominal TR approximately 3 ms. Semantics unreconciled; no conversion assumed. Exploratory. External VIF sensitivity excludes this field.'])
 wb.save(ROOT/'results/采集因素与新Rscore_完整结果.xlsx')
 fig,axes=plt.subplots(1,2,figsize=(12,6))
 for ax,cohort in zip(axes,['development','external_test']):
  sub=a[(a.analysis=='joint')&(a.cohort==cohort)&(a.outcome=='full_score')];y=np.arange(len(sub));ax.errorbar(sub.coefficient,y,xerr=[sub.coefficient-sub.CI_low,sub.CI_high-sub.coefficient],fmt='o',capsize=3);ax.axvline(0,color='gray',ls='--');ax.set(yticks=y,yticklabels=[LABEL[v] for v in sub.factor],xlabel='Adjusted coefficient [95% CI]',title=f'{cohort} (n={int(sub.N.iloc[0])})');ax.invert_yaxis()
 save(fig,'A1_Rscore_joint_associations')
 q=s[s.variable!='sex'].sort_values('SMD_external_minus_development');fig,ax=plt.subplots(figsize=(7,6));ax.scatter(q.SMD_external_minus_development,np.arange(len(q)));ax.axvline(0,color='gray',ls='--');ax.set(yticks=np.arange(len(q)),yticklabels=q.variable,xlabel='Standardized mean difference: external − development');save(fig,'A2_CMR_cohort_shift')
 fig,ax=plt.subplots(figsize=(7,3.5));ax.errorbar(cmp.Delta_AUC_B_minus_A,np.arange(len(cmp)),xerr=[cmp.Delta_AUC_B_minus_A-cmp.Delta_CI_low,cmp.Delta_CI_high-cmp.Delta_AUC_B_minus_A],fmt='o',capsize=4);ax.axvline(0,color='gray',ls='--');ax.set(yticks=np.arange(len(cmp)),yticklabels=cmp.model,xlabel='AUC difference: new − archived [95% CI]');save(fig,'A3_same_label_model_comparison')
 doc=Document();st=doc.styles['Normal'];st.font.name='Times New Roman';st.font.size=Pt(12);st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');doc.add_heading('采集因素与版本统一：新增实验中文回复',0)
 doc.add_paragraph('2026-11-10。针对编辑要求补充上一轮Rscore技术因素回归，本轮以新全流程模型的完整及无形状Rscore重做关联分析，并补充常规MRI参数的队列分布差异。所有分析均为已有队列的探索性补充，不改变既有模型和标签。')
 doc.add_paragraph('已复核1050例开发与309例外部患者的代表cine DICOM；1359例序列UID均与原记录一致，新增提取了接收带宽和重建视野。联合模型因层间距缺失，完整病例数为开发1040、外部300。另有9例开发患者没有本批cine元数据，不插补其采集参数。22例中心间距小于层厚，作为可能重叠或记录问题保留核查，未把中心间距误称为层间隙。')
 doc.add_paragraph('参数口径核查：外部DICOM记录的RepetitionTime中位数为43.94 ms，与稿件Table S1约3 ms的名义TR不同，目前不能确认二者物理含义一致。因此表图明确标为DICOM-reported TR，未擅自换算或替换；涉及该字段的分析作为探索性证据。外部VIF敏感性模型已移除该字段。')
 doc.add_paragraph('统计方法：按队列分别进行HC3稳健标准误OLS回归，调整LGE状态，开发集另调整外层折。单因素模型在可估计时另调整场强；联合模型同时纳入技术参数。连续效应按分析集参数标准差、评分按开发OOF评分标准差表达；按队列、评分和分析类型分别作Holm校正。接收带宽和重建FOV均有观测值，但后者不等同于实际处方FOV。')
 for _,r in diag.iterrows():doc.add_paragraph(f'{r.cohort} / {r.outcome}：n={int(r.N)}，技术参数整体Wald检验P={r.technical_omnibus_P:.3g}，联合模型R²={r.R2:.3f}（包含LGE及折效应，不能解释为纯技术因素解释率）。')
 doc.add_paragraph('完整Rscore的开发集联合模型中，层中心间距每增加1个标准差，对应评分增加0.305个开发OOF标准差（95% CI 0.174–0.435，Holm P约0.00004）。外部联合模型的单项参数校正后均未达到明确证据，但存在较强共线性，因此不能据此认定不存在采集影响。')
 doc.add_paragraph('已补按VIF>10逐步移除最高VIF技术变量的敏感性分析，保持相同完整病例且不删除LGE或折指标。开发集移除接收带宽，外部移除TR和重建FOV；结果及每步移除依据完整保存在Excel。此分析是共线性诊断补充，主分析不变，不按显著性筛选结果。')
 doc.add_paragraph('常规MRI队列差异：SRs的外部减开发标准化均值差为0.520，SRa为−0.465，SRe为−0.462，LAs为0.423。说明部分应变指标的分布确有差异，但不能仅凭分布差异断言是设备造成，或认定它解释了全部外部校准偏差。')
 doc.add_heading('同患者、同标签的新旧模型比较',1)
 for _,r in cmp.iterrows():doc.add_paragraph(f'{r.model}：存档概率在当前标签下的AUC为{r.archived_probability_AUC_current_labels:.4f}，新模型为{r.new_probability_AUC_current_labels:.4f}；配对差{r.Delta_AUC_B_minus_A:.4f}（95% CI {r.Delta_CI_low:.4f}至{r.Delta_CI_high:.4f}）。')
 doc.add_paragraph('三个配对差的区间均跨0，未观察到明确AUC差异，但不等于证明等效。存档Rscore在当前标签下AUC 0.847987，数值上对应稿件0.848；存档CatBoost联合模型重算为0.870696，仍未精确对应稿件0.875。后者历史来源问题不能用这次配对比较直接宣布解决。')
 doc.add_paragraph('由此，本轮可回复采集参数与Rscore关联的要求，并补强新旧结果版本比较。逐例LGE负荷、位置和模式、真实B/C中心映射、伦理和历史外部使用记录不能由这些统计反推出，暂不虚构。')
 for name,cap in [('A1_Rscore_joint_associations','联合采集参数回归，误差线为HC3 95%区间。'),('A2_CMR_cohort_shift','原始CMR参数的队列标准化均值差。'),('A3_same_label_model_comparison','同309例、同当前标签的新旧配对AUC差。')]:doc.add_picture(str(ROOT/f'figures/{name}.png'),width=Inches(6.2));doc.add_paragraph(cap)
 doc.save(ROOT/'response/补充采集分析_Response_中文版.docx');(ROOT/'response/补充采集分析_Response_中文版.txt').write_text('\n'.join(p.text for p in doc.paragraphs))
 active=CORE/'response/JMRI-26-1970_Response_中文版.txt';backup=active.with_name(active.stem+'_采集分析更新前备份_2026-11-10.txt')
 if not backup.exists():shutil.copy2(active,backup)
 tag='【2026-11-10完整采集因素分析已完成】'
 if tag not in active.read_text():active.write_text(tag+'\n新版完整/无形状Rscore对TR、TE、翻转角、层厚、层中心间距、场强、分辨率、带宽和重建视野的分析已补齐；含共线性敏感性分析及CMR队列差异。存档联合模型与新模型在同标签下AUC为0.8707/0.8698，配对差−0.0009（95% CI −0.0157至0.0140）。详见2026.11.06/acquisition_followup。此前“新版完整采集参数回归待补”现已完成。\n\n'+active.read_text())
 (ROOT/'audit/report_complete.json').write_text(json.dumps({'complete':True,'tables':len(t.index),'figures':3},indent=2));print('Report complete:',len(t.index),'tables, 3 figures')
if __name__=='__main__':main()
