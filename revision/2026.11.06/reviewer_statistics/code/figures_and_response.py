from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from docx import Document
from docx.shared import Pt,Inches
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
LABELS={'Rscore_full':'Full Rscore','Rscore_texture':'Non-shape Rscore','CMR':'MRI only','CMR_Rscore_full':'MRI + full Rscore','CMR_Rscore_texture':'MRI + non-shape Rscore','Selected':'Inner-selected strategy'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def read(name):return pd.read_csv(ROOT/f'tables/{name}.csv')
def save(fig,name):
 fig.tight_layout()
 for ext in ['png','pdf','tiff']:fig.savefig(ROOT/f'figures/{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

def main():
 data=read('predictions_external_test');metrics=read('performance');external=metrics.query("cohort=='external_test'").set_index('model')
 fig,axes=plt.subplots(1,2,figsize=(10,4.2))
 for ax,names in zip(axes,[['Rscore_full','Rscore_texture'],['CMR','CMR_Rscore_full','CMR_Rscore_texture']]):
  for name in names:
   fpr,tpr,_=roc_curve(data.event,data[name]);ax.plot(fpr,tpr,label=f"{LABELS.get(name,name)}: {external.loc[name,'AUC']:.3f}")
  ax.plot([0,1],[0,1],'k--',lw=.8);ax.set(xlabel='1 - specificity',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7,loc='lower right')
 axes[0].set_title('Radiomics score alone');axes[1].set_title('Incremental discrimination')
 save(fig,'R1_shape_ablation_ROC')
 p=read('prevalence_scenarios');fig,ax=plt.subplots(figsize=(6.4,4.4))
 for model,sub in p.groupby('model',sort=False):
  line=ax.plot(sub.prevalence,sub.NPV,label=LABELS.get(model,model))[0];ax.fill_between(sub.prevalence,sub.NPV_CI_low,sub.NPV_CI_high,color=line.get_color(),alpha=.12)
 ax.set(xlabel='Assumed LGE prevalence',ylabel='Negative predictive value',ylim=(0,1));ax.legend(fontsize=8);save(fig,'R2_NPV_prevalence')
 nr=read('NRI_IDI').query("cohort=='external_test' and B=='CMR_Rscore_full'")
 sub=nr[nr.metric.str.startswith('NRI')];fig,ax=plt.subplots(figsize=(6,4))
 ax.errorbar(sub.estimate,np.arange(len(sub)),xerr=[sub.estimate-sub.CI_low,sub.CI_high-sub.estimate],fmt='o',capsize=4);ax.axvline(0,color='gray',ls='--')
 ax.set(yticks=np.arange(len(sub)),yticklabels=sub.metric.map({'NRI_event':'Event NRI','NRI_nonevent':'Non-event NRI','NRI_total':'Total NRI'}),xlabel='Continuous NRI [95% CI]');save(fig,'R3_NRI_components')
 sg=read('subgroup_performance').query("model=='CMR_Rscore_full' and factor!='vendor'")
 fig,ax=plt.subplots(figsize=(7,4));labels=[(f'Field strength: {r.group} T' if r.factor=='field_strength_t' else f'Pixel spacing: {r.group}'.replace('<=','≤'))+f' (n={int(r.N)})' for _,r in sg.iterrows()]
 ax.errorbar(sg.AUC,np.arange(len(sg)),xerr=[sg.AUC-sg.AUC_CI_low,sg.AUC_CI_high-sg.AUC],fmt='o',capsize=4)
 ax.set(yticks=np.arange(len(sg)),yticklabels=labels,xlabel='External AUC [95% CI]',xlim=(.4,1));save(fig,'R4_acquisition_subgroups')
 joined=read('metadata_joined').query("cohort=='external_test'");fig,ax=plt.subplots(figsize=(6,4))
 groups=[s.Rscore_z.to_numpy() for _,s in joined.groupby('field_strength_t')];labels=[f'{g:g} T (n={len(s)})' for g,s in joined.groupby('field_strength_t')]
 ax.boxplot(groups,tick_labels=labels,showfliers=False);ax.set(ylabel='Rscore (development OOF scale)',title='External Rscore by field strength');save(fig,'R5_Rscore_field_strength')
 if (ROOT/'tables/segmentation_validation.csv').exists():
  seg=read('segmentation_validation');fig,ax=plt.subplots(figsize=(7,4));centers=['Overall','Anzhen','Renji','Shengyi'];x=np.arange(4)
  for j,(run,sub) in enumerate(seg.groupby('run',sort=False)):
   vals=sub.set_index('center').loc[centers];ax.bar(x+(j-.5)*.35,vals.pooled_pixel_Dice,width=.35,label=run)
  ax.set(xticks=x,xticklabels=centers,ylabel='Pooled-pixel validation Dice',ylim=(0,1));ax.legend(fontsize=8);save(fig,'R6_segmentation_validation')
 doc=Document();style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(12);style.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 doc.add_heading('JMRI-26-1970 补充统计与ROI证据：中文回复',0)
 doc.add_paragraph('本轮仅使用已完成实验的锁定预测和现有资料，不新增大规模训练。下列结果对应1059例开发集与309例外部队列；mViT及采集元数据分析另注明1050例匹配开发集。全文中的CatBoost分支与内层选择分类器策略分别报告，避免混用阈值。')
 pairs=read('shape_comparisons').query("cohort=='external_test'")
 full=external.loc['Rscore_full'];texture=external.loc['Rscore_texture'];combined=external.loc['CMR_Rscore_full']
 doc.add_heading('R3：形状特征与非形状特征的贡献',1)
 doc.add_paragraph(f'我们在任何监督筛选之前移除13个形状候选特征，使用其余474个一阶及纹理特征重新建模。完整Rscore和无形状Rscore单独的外部AUC分别为{full.AUC:.3f}和{texture.AUC:.3f}。')
 for _,r in pairs.iterrows():doc.add_paragraph(f'{r.B}相对{r.A}的ΔAUC为{r.Delta_AUC_B_minus_A:.3f}（95% CI {r.Delta_CI_low:.4f}至{r.Delta_CI_high:.4f}；Holm P={r.Holm_P:.4g}）。')
 doc.add_paragraph('结果表明，形状特征有额外贡献，但去除显式形状特征后，评分在传统MRI指标之外仍提供判别增量。该结果并不直接证明增量全部来源于微观纤维化或组织纹理机制。详见Table R1–R2。')
 op=read('operating_changes').query("B=='CMR_Rscore_full'").set_index('metric');base=external.loc['CMR']
 doc.add_heading('R1：特异度和错误分类变化',1)
 doc.add_paragraph(f'在各自开发数据确定的阈值下，加入完整Rscore后，外部假阳性由{int(base.FP)}例降至{int(combined.FP)}例，假阴性由{int(base.FN)}例降至{int(combined.FN)}例。特异度由{base.Specificity:.3f}升至{combined.Specificity:.3f}；敏感度由{base.Sensitivity:.3f}升至{combined.Sensitivity:.3f}。')
 doc.add_paragraph(f'对LGE阴性患者，7例由错误转为正确、2例由正确转为错误；精确配对检验P={op.loc["Specificity","McNemar_exact_P"]:.3f}。因此可以报告观察到的特异度改善，但不能据此声称统计学证实主要改善阴性识别，也不能将两个不同敏感度的操作点视为同等敏感度比较。详见Table R3。')
 doc.add_heading('R3：NRI分解与患病率敏感性',1)
 for _,r in nr.iterrows():doc.add_paragraph(f'{r.metric}：{r.estimate:.3f}（95% CI {r.CI_low:.3f}至{r.CI_high:.3f}）。')
 doc.add_paragraph('连续NRI分为事件与非事件两部分，并报告逐例概率上升、下降及不变的人数。总NRI可超过1，不应解释为超过100%的患者获得临床获益。另在固定外部敏感度与特异度的假设下报告30%–70%患病率情景的NPV及PPV，结果不是在新低患病率队列中的实测性能。详见Table R4–R5。')
 doc.add_heading('编辑、R1和R3：采集异质性',1)
 doc.add_paragraph('已补充最终模型及Rscore的场强、分辨率分层表现，校准及Rscore关联分析；各亚组沿用开发集阈值。像素间距定义为行、列间距的几何均值，分组界值取开发集的中位数。关联分析调整LGE状态、场强及像素间距，开发集另调整外层折。详见Table R6–R9。')
 doc.add_paragraph('厂商字段含匿名化值，且原始提取值与稿件外部扫描仪描述不一致；厂商结果仅保存在统计明细中，待核对后用于正式结论。外部中心映射尚缺，未将场强当作中心。')
 doc.add_heading('R3：心肌区域约束与分割质量',1)
 doc.add_paragraph('已完成的Mask引导ROI裁剪mViT外部AUC为0.793，匹配整图对照为0.797，差值−0.0035（95% CI −0.0337至0.0261）；未显示明确ROI收益。原投稿0.802作为历史参考单列，不将跨流程差异全部归因于Mask。详见Table R10。')
 if (ROOT/'tables/segmentation_validation.csv').exists():
  doc.add_paragraph('已核实分割项目使用三个私有中心的可靠人工标注帧，按患者划分952例训练、106例验证。发现两套正式运行：unet2d_myo_all_centers第5轮总体验证Dice 0.913；reproduce_cmr第13轮总体验证Dice 0.920。106例参与模型选择，故称患者级留出验证，不能称未参与模型选择的独立测试。Table R11分别报告两套来源，不以较高值替代当前Mask来源核查。')
  doc.add_paragraph('当前ROI Mask对应哪套权重、分割训练患者与本研究的跨数据集身份关系仍需确认。匿名编号同名不等于同一患者。已有记录可补充分割验证证据，但不能据此宣布外部独立分割精度已经验证。')
 doc.add_heading('临床资料交接',1)
 doc.add_paragraph('已导出309例外部病例核查表，以及CatBoost分支17例／Selected策略18例假阴性的合并核查名单。需要临床补齐LGE质量百分比、位置、模式、孤立插入点强化、中心及相关风险资料。指定数据目录保留影像，但未找到对应定量临床记录；不能从二分类event反推这些字段。')
 for image in sorted((ROOT/'figures').glob('*.png')):doc.add_picture(str(image),width=Inches(6.1))
 doc.save(ROOT/'response/补充统计_Response_中文版.docx')
 text='\n'.join(p.text for p in doc.paragraphs)
 (ROOT/'response/补充统计_Response_中文版.txt').write_text(text)
 print('Figures and Chinese response written.')

if __name__=='__main__':main()
