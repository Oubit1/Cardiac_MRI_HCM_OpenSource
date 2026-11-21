from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from docx import Document
from docx.shared import Pt,Inches
from docx.oxml.ns import qn
from openpyxl import load_workbook
from openpyxl.styles import Font,Alignment,Border,Side
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(CORE/'reviewer_statistics/code'))
from manuscript_tables import Tables,ci,num
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42})
def main():
 assert json.loads((ROOT/'audit/complete.json').read_text())['complete']
 m=pd.read_csv(ROOT/'results/metrics.csv');p=pd.read_csv(ROOT/'results/paired_loss_changes.csv');names={'Rscore_CMR_CatBoost':'CMR + Rscore (CatBoost)','Rscore_CMR_EN':'CMR + Rscore (elastic net)'}
 t=Tables();t.add('Table C3','Logistic recalibration evaluated within outer folds and externally',['Cohort / model / state','AUC','Calibration intercept [95% CI]','Calibration slope [95% CI]','Brier','Log loss'],[[f'{r.cohort} / {names[r.model]} / {r.state}',num(r.AUC),ci(r,'Calibration_intercept_joint'),ci(r,'Calibration_slope_joint'),num(r.Brier),num(r.Log_loss)] for _,r in m.iterrows()], 'Calibration intercept and slope are jointly estimated. Positive-slope logistic calibration was fitted only on development-side predictions. Outer evaluation uses five separate calibrators; pooled development AUC can change across folds. External AUC is invariant under the final monotonic mapping. Bootstrap CIs condition on fitted predictions; the previously evaluated external cohort is reused.',widths=[7.1,2.3,5.2,5.2,3.1,3.2])
 t.add('Table C4','Paired changes in probabilistic prediction error',['Cohort','Model','Metric','Calibrated − raw [95% CI]'],[[r.cohort,names[r.model],r.metric,f'{r.calibrated_minus_raw:.5f} [{r.CI_low:.5f}–{r.CI_high:.5f}]'] for _,r in p.iterrows()], 'Negative values favor recalibration. Intervals use 2000 paired, outcome-stratified patient bootstrap samples. No model or calibration method was selected using these external comparisons. This is a post hoc model-update analysis motivated by previously observed external calibration.')
 t.doc.save(ROOT/'results/重新校准结果_稿件样式.docx');t.wb.save(ROOT/'results/重新校准结果_稿件样式.xlsx')
 with pd.ExcelWriter(ROOT/'results/重新校准完整结果.xlsx') as writer:
  for name,file in [('Metrics','results/metrics.csv'),('Paired_changes','results/paired_loss_changes.csv'),('Calibrators','results/calibrators.csv'),('Development_predictions','results/development_predictions.csv'),('External_predictions','results/external_predictions.csv'),('Inner_predictions','audit/calibration_training_inner_predictions.csv'),('Partition_checks','audit/partition_checks.csv')]:pd.read_csv(ROOT/file).to_excel(writer,sheet_name=name,index=False)
 wb=load_workbook(ROOT/'results/重新校准完整结果.xlsx')
 for ws in wb:
  ws.sheet_view.showGridLines=False;ws.freeze_panes='A2';last=ws.max_row
  for row in ws:
   for c in row:c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.alignment=Alignment(wrap_text=True);c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=22
 wb.save(ROOT/'results/重新校准完整结果.xlsx')
 bins=[]
 for cohort,file in [('development','development_predictions'),('external','external_predictions')]:
  d=pd.read_csv(ROOT/f'results/{file}.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.5))
  for ax,(name,label) in zip(axes,names.items()):
   for state,color in [('raw','#444444'),('calibrated','#0072B2')]:
    col=name if state=='raw' else name+'_calibrated';obs,pred=calibration_curve(d.event,d[col],n_bins=8,strategy='quantile');ax.plot(pred,obs,'o-',label=state.capitalize(),color=color)
    bins.extend([{'cohort':cohort,'model':name,'state':state,'mean_prediction':a,'observed_fraction':b} for a,b in zip(pred,obs)])
   ax.plot([0,1],[0,1],'--',color='gray',label='Ideal');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed LGE fraction',title=label);ax.legend()
  fig.suptitle('Outer-fold evaluation' if cohort=='development' else 'Reused external cohort');fig.tight_layout()
  for ext in ['png','pdf','tiff']:fig.savefig(ROOT/f'figures/recalibration_{cohort}.{ext}',dpi=300,bbox_inches='tight')
  plt.close(fig)
 pd.DataFrame(bins).to_csv(ROOT/'results/calibration_bins.csv',index=False)
 doc=Document();st=doc.styles['Normal'];st.font.name='Times New Roman';st.font.size=Pt(12);st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体');doc.add_heading('概率重新校准：补充实验与中文回复',0)
 doc.add_paragraph('2026-11-10。本分析在观察到既有外部校准偏差后开展，属于模型更新的补充分析。主要模型固定为CMR＋Rscore CatBoost，Elastic-net作为参照；预先固定采用无类别加权的Logistic校准，不按外部结果挑选方法。')
 doc.add_paragraph('方法：在每个外层训练集中，复用原内层五折独立建立的CMR筛选、Radiomics筛选及Rscore交叉拟合数据，重新拟合下游分类器并生成内层留出预测，再拟合正斜率Logistic校准器。原先仅由外层训练患者选出的CatBoost参数在该折内部复用；内层预测对这些已选参数而言为交叉拟合，不声称参数选择在每个校准内折再次嵌套。外层验证患者不参与上述任何步骤，最后评价其校准后概率。')
 doc.add_paragraph('最终校准器仅用1059例开发集外层OOF预测和结局拟合，锁定后用于已有最终模型对309例外部患者的概率。不同训练规模和所选参数可能影响校准迁移。外部结局未参与校准器拟合。保持原分类决定：将原概率阈值同步映射，而不是在外部重新选阈值。')
 for name,label in names.items():
  doc.add_heading(label,1)
  for cohort in ['development','external']:
   z=m[(m.model==name)&(m.cohort==cohort)].set_index('state');a=z.loc['raw'];b=z.loc['calibrated']
   doc.add_paragraph(f'{cohort}：AUC {a.AUC:.3f}→{b.AUC:.3f}；校准截距 {a.Calibration_intercept_joint:.3f}→{b.Calibration_intercept_joint:.3f}；斜率 {a.Calibration_slope_joint:.3f}→{b.Calibration_slope_joint:.3f}；Brier {a.Brier:.4f}→{b.Brier:.4f}；Log loss {a.Log_loss:.4f}→{b.Log_loss:.4f}。95%区间见Table C3。')
  for _,r in p[(p.model==name)&(p.cohort=='external')].iterrows():doc.add_paragraph(f'外部{r.metric}配对变化（校准后减校准前）为{r.calibrated_minus_raw:.5f}，95% CI {r.CI_low:.5f}至{r.CI_high:.5f}；负值代表改善。')
 doc.add_paragraph('本次结果：CatBoost外部斜率由0.531升至0.586，但校准后95%区间仍为0.450–0.790，未达到理想校准。外部Log loss下降0.01756（95% CI −0.02993至−0.00655），提示该指标改善；Brier差值区间跨0，未显示明确改善。Elastic-net参照的外部Log loss上升0.01382（0.00446至0.02410），没有受益。开发CatBoost原斜率约0.906，而外部约0.531，提示开发数据学到的较小幅度修正不足以消除外部偏差；仅凭此结果不能判定偏差的具体成因。')
 doc.add_paragraph('校准后外部AUC和同步映射阈值下的分类结果保持不变。该方法只更新概率表达，不会自动减少假阴性。建议将该结果作为校准更新的敏感性分析，保留原始主结果，报告部分概率损失改善及残余偏差；不能宣称已经实现良好校准，也不能将校准拟合集上的截距0、斜率1作为验证成功的证据。')
 for cohort in ['development','external']:doc.add_picture(str(ROOT/f'figures/recalibration_{cohort}.png'),width=Inches(6.2));doc.add_paragraph(f'校准曲线（{cohort}）：按预测概率分为8个分位数组，显示各组平均预测和实际LGE比例。连线用于描述；校准截距和斜率来自患者级Logistic回归，并非折线段斜度。')
 doc.save(ROOT/'response/重新校准_Response_中文版.docx');text='\n'.join(p.text for p in doc.paragraphs);(ROOT/'response/重新校准_Response_中文版.txt').write_text(text)
 active=CORE/'response/JMRI-26-1970_Response_中文版.txt';backup=active.with_name(active.stem+'_重新校准前备份_2026-11-10.txt')
 if not backup.exists():shutil.copy2(active,backup)
 tag='【2026-11-10重新校准实验】';old=active.read_text()
 if tag not in old:
  z=m[(m.model=='Rscore_CMR_CatBoost')&(m.cohort=='external')].set_index('state');a=z.loc['raw'];b=z.loc['calibrated'];intro=f'{tag}\n已完成外层隔离的概率校准及开发集拟合后外部评价。CatBoost外部斜率{a.Calibration_slope_joint:.3f}→{b.Calibration_slope_joint:.3f}；Brier {a.Brier:.4f}→{b.Brier:.4f}；Log loss {a.Log_loss:.4f}→{b.Log_loss:.4f}；AUC保持{a.AUC:.3f}。详细结果见2026.11.06/recalibration，属于既有外部队列上的模型更新分析。\n\n';active.write_text(intro+old)
 print(m[['cohort','model','state','AUC','Calibration_slope_joint','Brier','Log_loss']].to_string(index=False));print(p.to_string(index=False))
if __name__=='__main__':main()
