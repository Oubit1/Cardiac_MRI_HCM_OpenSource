"""Combine follow-up tables/reply and record provenance without changing training outputs."""
from pathlib import Path
from copy import deepcopy,copy
import json,shutil,zipfile
import numpy as np,pandas as pd
from docx import Document
from docx.shared import Inches
from openpyxl import load_workbook
from openpyxl.styles import Font,Border,Side,Alignment,PatternFill
from manuscript_tables import Tables
ROOT=Path(__file__).resolve().parents[1]
def main():
 assert json.loads((ROOT/'audit/mask_extended_reproduction.json').read_text())['completed']
 match=pd.read_csv(ROOT/'audit/mask_extended_reproduction.csv');assert len(match)==24
 audit=json.loads((ROOT/'audit/ROI_fallback_criteria_audit.json').read_text());assert audit['same_failure_classification']==1123
 evidence=[]
 for run,g in match.groupby('run',sort=False):evidence.append(f'{run}：{g.agreement_Dice_to_existing_mask.min():.6f}–{g.agreement_Dice_to_existing_mask.max():.6f}，逐体完全一致{int(g.exact_match.sum())}/{len(g)}')
 source_note='Mask来源补充核查：按患者ID等距抽取开发6例、外部6例的中间切片，对两个候选检查点按源代码预处理、0.5阈值、CPU float32重新推理。与现有Mask的Dice一致度为：'+'；'.join(evidence)+'。这属于输出一致性核查，不是与人工参考比较的精度；12例一致性结果强烈支持现有Mask来自unet2d_myo_all_centers第5轮检查点，其已有患者级选模验证Dice为0.913。原始逐例推理日志仍未找到，故将此归为抽样复现支持的来源判断。'
 source=(ROOT/'audit/segmentation_provenance.json');prov=json.loads(source.read_text());prov['candidate_reproduction_evidence']={'source':str(ROOT/'audit/mask_extended_reproduction.csv'),'volumes':12,'reference_accuracy':False,'summary':evidence};source.write_text(json.dumps(prov,ensure_ascii=False,indent=2))
 failure=pd.read_csv(ROOT/'tables/ROI_fallback_criteria_summary.csv');t=Tables();labels={'empty_frame':'At least one empty phase','area_fraction_outside_range':'Median mask fraction outside 0.2–30%','median_components_gt2':'Median 4-connected components >2','median_temporal_dice_lt05':'Median adjacent-phase Dice <0.5'}
 rows=[]
 for flag,label in labels.items():
  g=failure[failure.criterion==flag].set_index('cohort');rows.append([label,*[f'{int(g.loc[c,"flagged_slices"])} ({100*g.loc[c,"fraction_of_fallback_slices"]:.2f})' for c in ['development','external_test']]])
 t.add('Table R14','Recomputed triggers among recorded fallback slices',['Reference-free criterion','Development (n=772), n (%)','External (n=351), n (%)'],rows,'Criteria overlap; percentages need not sum to 100%. All 1123 current masks reproduce the recorded fallback classification. These are preprocessing triggers, not measures of manual-reference accuracy. Small extra components may trigger fallback without establishing that the main myocardium contour is clinically incorrect. No masks, thresholds or model weights were changed.')
 combined=Document(ROOT/'tables/补充统计表_稿件样式.docx')
 for other in [Document(ROOT/'tables/ROI追加表_稿件样式.docx'),t.doc]:
  combined.add_page_break()
  for node in other.element.body:
   if not node.tag.endswith('sectPr'):combined.element.body.insert(-1,deepcopy(node))
 assert len(combined.tables)==14;combined.save(ROOT/'tables/全部补充表_R1-R14_稿件样式.docx')
 # Complete machine-readable precision, plus human-readable manuscript-style follow-up tables.
 wb=load_workbook(ROOT/'tables/统计明细.xlsx')
 additions={'ROI患者汇总':'tables/ROI_patient_audit.csv','ROI回退汇总':'tables/ROI_fallback_summary.csv','ROI分层性能':'tables/ROI_fallback_performance.csv','ROI配对比较':'tables/ROI_fallback_paired.csv','ROI回退触发条件':'tables/ROI_fallback_criteria_summary.csv','分割逐轮记录':'tables/Segmentation_epoch_history.csv','外部中心字段核查':'tables/external_center_metadata_audit.csv','Mask候选复现':'audit/mask_extended_reproduction.csv'}
 for sheet,file in additions.items():
  if sheet in wb:del wb[sheet]
  ws=wb.create_sheet(sheet);df=pd.read_csv(ROOT/file).replace([np.inf,-np.inf],np.nan).fillna('');ws.append(list(df.columns))
  for row in df.itertuples(index=False,name=None):ws.append(list(row))
  ws.sheet_view.showGridLines=False;ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions;last=ws.max_row
  for row in ws:
   for c in row:c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.alignment=Alignment(wrap_text=True);c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=23
 wb.save(ROOT/'tables/统计明细.xlsx')
 pub=load_workbook(ROOT/'tables/补充统计表_稿件样式.xlsx')
 for book in [load_workbook(ROOT/'tables/ROI追加表_稿件样式.xlsx'),t.wb]:
  for ws in book:
   out=pub.create_sheet(ws.title)
   for row in ws:
    for cell in row:
     z=out.cell(cell.row,cell.column,cell.value)
     for prop in ['font','fill','border','alignment','protection']:setattr(z,prop,copy(getattr(cell,prop)))
     z.number_format=cell.number_format
   for merged in ws.merged_cells.ranges:out.merge_cells(str(merged))
   for key,dim in ws.column_dimensions.items():out.column_dimensions[key].width=dim.width
   for key,dim in ws.row_dimensions.items():out.row_dimensions[key].height=dim.height
   out.sheet_view.showGridLines=False;out.freeze_panes='B3'
 pub.save(ROOT/'tables/全部补充表_R1-R14_稿件样式.xlsx')
 doc=Document(ROOT/'response/补充统计_Response_中文版.docx')
 for p in doc.paragraphs:
  if p.text.startswith('当前ROI Mask对应哪套权重'):p.text='Mask来源的最新复现核查见文末追加结果；分割训练患者与本研究的跨数据集身份关系仍需确认。'
 doc.add_page_break();doc.add_heading('追加：ROI回退、分割来源及中心字段核查',1)
 follow=Document(ROOT/'response/ROI追加回复_中文版.docx')
 for p in follow.paragraphs:
  if p.text and not p.text.startswith('Figure R'):doc.add_paragraph(p.text)
 doc.add_paragraph(source_note)
 center=json.loads((ROOT/'audit/external_center_metadata_audit.json').read_text())
 doc.add_paragraph(f'已抽查309例外部患者的{center["headers_sampled"]}份DICOM头，包括原代表序列及每例最多16份等距抽样文件；未发现可用机构名称。此检查不是穷尽全部DICOM头，暂不能建立B/C中心映射。非代表序列中的Manufacturer也可能属于后处理软件，不能直接作为扫描仪厂商分组。')
 doc.add_paragraph('回退原因复核：开发772张回退切片中700张、外部351张中339张触发连通区域中位数大于2；不同规则可同时触发。全部1123张现有Mask与缓存记录的回退分类一致。这里只报告预处理规则，不把小碎片触发直接解释为心肌主轮廓错误。Table R14记录各项触发数量。')
 for name,caption in [('R7_segmentation_learning_curves','Figure R7. 两套正式分割运行的训练损失与逐中心验证Dice；虚线标示已保存检查点选择轮次。'),('R8_ROI_actual_inputs','Figure R8. 第0帧原图、现有Mask与裁剪框、整图模型输入、ROI模型输入；回退切片的两种输入完全相同。'),('R9_ROI_fallback_paired','Figure R9. 全外部队列及按回退状态划分的探索性配对AUC差（ROI减整图）与95%区间。')]:
  doc.add_picture(str(ROOT/f'figures/{name}.png'),width=Inches(6.2));doc.add_paragraph(caption)
 doc.save(ROOT/'response/综合补充回复_中文版.docx');(ROOT/'response/综合补充回复_中文版.txt').write_text('\n'.join(p.text for p in doc.paragraphs))
 active=ROOT.parent/'response/JMRI-26-1970_Response_中文版.txt';backup=active.with_name(active.stem+'_ROI追加前备份_2026-11-09.txt')
 if not backup.exists():shutil.copy2(active,backup)
 old=active.read_text()
 if old.startswith('2026-11-09 ROI追加核查已完成：'):old=old.split('\n\n',1)[1]
 update='2026-11-09 ROI追加核查已完成：新增Table R12–R14及Figure R7–R9；综合回复与全部14张表位于reviewer_statistics。外部198/309例有至少一张回退切片，111例没有回退；分层配对差的区间均跨0。1123张回退切片复核全部一致。5203份外部DICOM头抽样未找到可用机构名称。\n'+source_note+'\n\n'
 active.write_text(update+old)
 (ROOT/'audit/followup_delivery_validation.json').write_text(json.dumps({'passed':True,'tables':14,'figure_count':9,'external_patients':309,'fallback_rechecks':1123,'candidate_reproduction_volumes':12,'candidate_checkpoints':2,'training_outputs_unchanged':True},indent=2))
 print(source_note)
if __name__=='__main__':main()
