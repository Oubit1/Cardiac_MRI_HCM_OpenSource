from pathlib import Path
import json,shutil,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,confusion_matrix
from openpyxl import Workbook
from openpyxl.styles import Font,Border,Side,Alignment
from docx import Document
ROOT=Path(__file__).resolve().parents[1]
def main():
 checks=[]
 metrics=pd.read_csv(ROOT/'tables/performance.csv');cuts=json.loads((ROOT/'inputs/thresholds.json').read_text())
 for cohort,n in [('development',1059),('external_test',309)]:
  d=pd.read_csv(ROOT/f'tables/predictions_{cohort}.csv');assert len(d)==n and d.ID.is_unique
  for _,r in metrics[metrics.cohort==cohort].iterrows():
   assert np.isclose(roc_auc_score(d.event,d[r.model]),r.AUC)
   if cohort=='external_test':
    tn,fp,fn,tp=confusion_matrix(d.event,d[r.model]>=cuts[r.model]).ravel()
    assert [tn,fp,fn,tp]==[r.TN,r.FP,r.FN,r.TP]
  checks.append(f'{cohort}: patient IDs, AUCs and external confusion matrices verified')
 seg=pd.read_csv(ROOT/'tables/segmentation_validation.csv');assert len(seg)==8
 doc=Document(ROOT/'tables/补充统计表_稿件样式.docx');assert len(doc.tables)==11
 for table in doc.tables:
  for row in table.rows:
   for cell in row.cells:
    for p in cell.paragraphs:
     for run in p.runs:assert run.font.name=='Times New Roman' and run.font.size.pt==12
 checks.append('11 Word tables: Times New Roman 12 pt verified')
 wb=Workbook();wb.remove(wb.active)
 for name,title in [('clinical_review_all_external','外部309例'),('false_negative_review','假阴性合并名单')]:
  df=pd.read_csv(ROOT/f'tables/{name}.csv').fillna('');ws=wb.create_sheet(title);ws.append(list(df.columns))
  for row in df.itertuples(index=False,name=None):ws.append(list(row))
  ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions;ws.sheet_view.showGridLines=False
  last=ws.max_row
  for row in ws:
   for c in row:
    c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.alignment=Alignment(wrap_text=True)
    c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=23
 wb.save(ROOT/'tables/临床资料待填核查表.xlsx')
 source=ROOT.parent/'response/JMRI-26-1970_Response_中文版.txt'
 backup=source.with_name(source.stem+'_补统计前备份_2026-11-09.txt')
 if not backup.exists():shutil.copy2(source,backup)
 s=backup.read_text()
 updates={
 '11':('单独Rscore统计已完成','完整Rscore外部AUC 0.844（95% CI 0.795–0.892），无形状Rscore 0.816（0.759–0.869）；配对差0.0286（0.0074–0.0523），Holm P=0.017。采用各折原始预测概率，未混合折内重新标准化分数。Table R1–R2。'),
 '12':('增量与形状配对比较已完成','无形状组合相对MRI-only增量0.0493（0.0254–0.0771）；完整组合相对无形状组合差0.0129（0.0002–0.0256），Holm P=0.037。Table R2。'),
 '17':('患病率情景统计及图表已完成','已补30%、40%、50%、60%、70%患病率的NPV、PPV及2000次患者分层bootstrap区间；固定外部敏感度、特异度的情景分析。Table R5、Figure R2。'),
 '20':('场强与分辨率已完成；中心与厂商来源待核查','已计算最终模型、单独Rscore的分层判别、分类指标及校准，使用原开发阈值。外部309例全部纳入。厂商存在匿名化值并与稿件描述不一致，其探索结果保存在明细；B/C中心映射未找到，未以场强替代中心。Table R6–R8。'),
 '21':('采集关联统计已完成；中心因素待补','已报告Rscore分布、连续像素间距关联及调整LGE/场强/像素间距的HC3回归，开发集另调整外层折。开发元数据覆盖1050/1059例，外部309/309例。中心不能调整，厂商结果暂作探索。Table R9及统计明细。'),
 '24':('NRI分解已完成','完整Rscore加入MRI后，外部事件NRI=0.594（0.481–0.698），非事件NRI=0.546（0.361–0.711），总NRI=1.141（0.930–1.339），IDI=0.153（0.116–0.193）。均为连续NRI定义；Table R4。'),
 '25':('分类变化配对统计已完成','外部假阳性33→28、假阴性31→17。阴性患者7例改善、2例变差，精确McNemar P=0.180；报告观察到的特异度改善，不将其写为显著或同敏感度比较。Table R3。')}
 import re
 for number,(status,detail) in updates.items():
  pat=rf'(?m)^{number}\. .*?(?=\n\d{{2}}\. |\Z)'
  match=re.search(pat,s,re.S)
  if match:
   block=match.group();heading=block.splitlines()[0];heading=re.sub(r'【.*?】',f'【{status}】',heading)
   # Preserve historical audit detail separately, show current result in the active checklist.
   section_tail='\n'.join(line for line in block.splitlines()[1:] if re.match(r'^[一二三四五六七八九十]+、',line))
   s=s[:match.start()]+heading+'\n本次更新：'+detail+'\n是否重训：否，使用已保存的预测完成。\n'+section_tail+'\n'+s[match.end():]
 s=s.replace('本次只整理任务，不启动新实验。','本轮已完成既有预测的补充统计，不新增模型训练。')
 s=s.replace('08. Radiomics 与 mViT 比较是否公平？【ROI实验部分回应】','08. Radiomics 与 mViT 比较是否公平？【ROI及分割验证材料已补，Mask权重关联待核查】\n本次补充：分割项目三中心952例训练、106例验证；两套运行总体验证Dice 0.913/0.920，按中心及患者汇总见Table R11。该验证集参与最佳epoch选择，不能标作独立测试。')
 intro='2026-11-09 补充统计更新：Table R1–R11及6幅图已生成，文件位于2026.11.06/reviewer_statistics。中文回复见response/补充统计_Response_中文版.docx。原清单已备份；以下11、12、17、20、21、24、25条已同步更新。\n分割证据：三个私有中心，952例训练、106例患者级验证。两次运行Dice为0.913及0.920，均用于模型选择，不能称独立测试。尚需关联实际HCM Mask使用的权重，并核实分割训练患者与本研究的身份交叉。Table R11列出两次运行及逐中心结果。\n仍需资料：B/C中心映射、LGE百分比/位置/模式、Mask推理来源；厂商字段与稿件描述需要核对。\n\n'
 source.write_text(intro+s)
 (ROOT/'audit/delivery_validation.json').write_text(json.dumps({'passed':True,'checks':checks,'figures':len(list((ROOT/'figures').glob('*.png'))),'tables':11},ensure_ascii=False,indent=2))
 print('Numerical validation passed; clinical workbook and active response checklist updated.')
if __name__=='__main__':main()
