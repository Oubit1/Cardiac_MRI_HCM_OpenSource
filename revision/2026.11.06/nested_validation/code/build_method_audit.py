"""Document-grounded audit; does not train or modify historical results."""
from settings import *
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROWS = [
('分析对象', '正文§Representation Benchmarking、§Final Model；Table 2/3',
 '正式稿为九种弹性网基准模型（1050/309）及最终CatBoost（1059/309）。',
 '当前新协议仅六种模型：CMR、Rscore、FT及其组合；单次五折；不含Frozen或CatBoost。',
 '可以作为FT相关补充验证；不能声称重做了Table 2全部九模型或Table 3。'),
('CMR筛选', '审稿编辑问题2；R3 major 6；正文§Predictor Selection',
 '原稿先在整个开发集筛选MWT、LAs、LVGRS、LVGLS。',
 '新代码从16个候选变量开始，在每个外层训练集内重新执行LASSO；内层预处理也重新拟合。',
 '方向正确。各折变量可不同，不能强制保留历史四变量；最终全开发集再筛选。'),
('Rscore与融合', '编辑问题2；26.9.19/Rscore_nested_OOF_elasticnet.R',
 '历史Rscore采用10折×5重复交叉拟合；已有全开发集OOF分数不自动满足下游外层独立性。',
 '新代码在每个外层训练集重新构建Rscore，融合训练分数在该训练集内五折交叉拟合；外层测试分数由外层训练集模型生成。',
 '可用于新增敏感性分析。五折、不重复及固定lambda网格属于协议改变，不能写成历史方法完全复现；不得套用历史23特征公式。'),
('mViT权重复用', '编辑问题2；五折checkpoint与split核验',
 '现有extended60五折训练ID与新外层划分一致，外层测试不参与该折训练。',
 '可复用对应折权重，分别提取840训练与210测试病例的特征。训练840例特征是拟合内表示，不是OOF。',
 '固定下游融合参数时可评价外层测试；若要在内层调融合超参数并声称内层全流程独立，须在内层重新训练监督编码器或采用独立的内层特征生成设计。'),
('融合超参数', '正文§Representation Benchmarking；E3.2/E3.3',
 'E3描述开发集内选择超参数；当前方案固定C=0.1、l1_ratio=0.5。',
 '固定融合参数能避免新增内层监督表示复用的问题，但已知历史开发/外部结果，不能称原研究预注册选择。',
 '写为修订阶段锁定的固定参数敏感性分析；统一分类器不等于每个模型均已充分优化。'),
('CatBoost重新优化', 'Model construction/CatBoost/CatBoost_final_training.py:181–189；metadata阈值0.492424',
 '代码先search.fit(X,y)，再对search.best_estimator_执行cross_val_predict；阈值吻合稿件0.492。',
 '这一实现未在外层训练折重新搜索参数；输入又是预先生成的Rscore和四变量。',
 '不能确认Table 3为全流程重复嵌套结果。需1059例独立分支重建；若评估五分类器选择策略，选择分类器也要进入外层训练集的内层流程。固定CatBoost敏感性分析不能追溯证明原选择过程无偏。'),
('CatBoost结果来源', 'CatBoost_final_results.xlsx；当前analysis_manifest；正式稿Table 3',
 '本地结果表开发AUC=0.917227，与主文吻合，但外部AUC=0.905174，未吻合0.875；其中6例外部标签与当前清单不同。',
 '按ID与当前清单匹配309例、保持旧概率不变重算：外部AUC=0.870696；不含Rscore为0.821825；混淆矩阵TN70/FP27/FN17/TP195，与稿件操作指标吻合。',
 '正式稿外部AUC的最终逐例概率来源仍待追溯。阈值及开发AUC吻合不足以证明是完全相同版本。不得为匹配0.875修改标签，也不能把旧表直接当作正式稿的唯一数据来源。'),
('Frozen分支', '正文§mViT Extraction和Table 2',
 'Frozen有512维表示及三种组合，是正文九模型的一部分。',
 '新六模型协议尚不包含Frozen。',
 '若完整更新Table 2应补三种Frozen模型；确认提取无结局适配后可复用Frozen特征，但标准化和分类器必须在折内拟合。'),
('训练时长实验', 'R1 major 1/3/4；已完成core24h',
 '已完成20/60 epoch上限比较、逐epoch训练验证曲线、五折refit及全开发集最终模型。',
 '可回答20 epoch是否限制这一训练方案；没有完成全面学习率搜索或多随机种子比较。',
 '不得写训练达到全局最优或证明20 epoch对所有设置足够；外部模型须注明全开发集重训、训练轮次及权重文件。'),
('时间采样描述', 'E3.1；code/train.py；vendor/legacy_trainer.py:124',
 'E3写uniform linear index interpolation；实际代码用linspace后round选取16帧。',
 '已有实验执行的是均匀索引采样，未对时间维做线性插值。',
 '应按实际代码修改E3；不要为迎合旧文字悄悄改变预处理并混用既有结果。'),
('ROI公平性', 'R3 major 5；已完成roi_matched_full',
 '已完成cine掩膜引导的ROI裁剪敏感性实验，含回退病例；不是手工心肌掩膜完全匹配。',
 '裁剪保留血池及部分邻近组织；掩膜来源与独立性证据尚不足以证明专家标注等价。',
 '作为ROI引导裁剪补充证据，报告回退数量与比例；不能写成完全消除了标注优势。'),
('阈值与外部使用', '编辑问题1；正文Table 3；新协议',
 '外部队列此前已查看，修订也已有结果；本轮无法恢复为首次盲测。',
 '新阈值仅用开发集OOF满足敏感度≥0.95时最大特异度选择，再用于外部。',
 '记录决策时间线；新外部结果称重复使用队列上的敏感性分析。不能把同批OOF选阈值后的敏感度当独立验证；不要强制沿用0.492到新模型。'),
('无形状Rscore', 'R3 major 4',
 '审稿要求去掉形状后重建，仅保留一阶和纹理，并评价CMR之外增量。',
 '新协议仍含487个ICC合格特征，尚未实现独立无形状分支。',
 '应在筛选、调参前去掉全部shape/shape2D特征，折内重建；不能仅将旧23特征公式的5个系数置零。'),
('校准/异质性/安全性', '编辑总评；R3 major 1/2/3/8；R1 major 2',
 '需外部校准截距/斜率及CI、中心/磁场/厂商/分辨率分析、NPV患病率曲线、假阴性LGE范围与模式。',
 'mViT补实验不自动回答最终CatBoost的这些问题。',
 '以相应锁定模型的逐例概率和真实临床/设备元数据计算；不能在同一外部队列重校准后将拟合内改善称独立验证；缺失临床结局不能推断ICD获益。'),
('文件版本与报告', '主文§Radiomics；paper/Supplementary materials.docx；编辑批注117',
 '主文765特征/ICC≥0.80；目录内旧补充材料写1145/ICC≥0.75及LightGBM；E3还有占位文字。',
 '目录内这些补充文件不能直接视为与本次主文同版；需按来源和实际配置核实后统一。',
 '以正式主文确定CatBoost角色；完整标记1059/1050、分类器和Rscore版本。按编辑要求收敛临床实施/省略对比剂结论。'),
]

def main():
 out=ROOT/'audit/manuscript_review';out.mkdir(parents=True,exist_ok=True)
 columns=['核查项','依据','正式稿或历史证据','当前状态','处理意见']
 df=pd.DataFrame(ROWS,columns=columns)
 with pd.ExcelWriter(out/'方法核查与实验对应表.xlsx',engine='openpyxl') as writer:
  df.to_excel(writer,index=False,sheet_name='方法核查')
  ws=writer.sheets['方法核查'];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
  from openpyxl.styles import Alignment,Font,PatternFill
  for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='17365D')
  for i,width in enumerate([23,48,65,65,75],1):ws.column_dimensions[chr(64+i)].width=width
  for row in ws.iter_rows(min_row=2):
   ws.row_dimensions[row[0].row].height=96
   for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='top')
 doc=Document();section=doc.sections[0];section.top_margin=section.bottom_margin=Cm(2)
 style=doc.styles['Normal'];style.font.name='Calibri';style.font.size=Pt(10)
 style.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
 doc.add_heading('JMRI-26-1970 实验方法核查',0)
 doc.add_paragraph('核查日期：2026-11-08。依据用户指定正式审稿意见、带修订主稿及56条批注，并核对本地代码与相关补充材料。本文是内部方法核查，不是已完成新实验的回复信。')
 doc.add_heading('结论',1)
 doc.add_paragraph('不能确认“所有方法均无问题”。当前同一五折中重建CMR、Rscore并复用对应折mViT的方向可行，但只能作为修订阶段固定融合参数的FT相关敏感性分析。它不覆盖主文全部九模型，也不证明1059例CatBoost重新优化的全过程已经嵌套。历史结果、已完成mViT补实验和待跑验证应分别标记。')
 for name,source,evidence,state,action in ROWS:
  doc.add_heading(name,2)
  for label,value in [('依据',source),('证据',evidence),('现状',state),('建议',action)]:doc.add_paragraph(f'{label}：{value}')
 doc.add_heading('执行顺序与可复用边界',1)
 for text in [
  '优先完成1050例、六模型、固定融合参数的共同外层五折敏感性分析。沿用既有五折划分和对应mViT权重，无需重新训练这五个编码器；GPU任务是提取特征，CMR/Rscore主要使用CPU。',
  '完整更新Table 2时，补入Frozen三模型，明确新分析的单次五折与历史重复嵌套的区别。不同外层模型的256维表示只在本折内训练融合，不能跨折拼接后再随机划分训练。',
  '若要声称全文最终模型也完成严格验证，另建1059例CatBoost分支：外层训练集内完成CMR筛选、Rscore构建、候选模型/参数选择，内层验证涉及的所有上游监督步骤也要重建。最终只在外层未参与拟合病例上评价；之后全开发集定模型、定阈值，再计算外部敏感性结果。',
  '无形状评分、最终模型外部校准/亚组以及假阴性临床资料是其他审稿问题的独立任务；训练时长和ROI结果不能代替它们。',
  '截至本核查，新验证尚无真实数据完整训练结果。已通过的输入划分核验和合成数据测试不等于临床实验完成。不承诺未经首折实测的运行时长。']:
  doc.add_paragraph(text,style='List Number')
 doc.add_heading('回复措辞边界',1)
 doc.add_paragraph('现阶段可以写：“我们识别到原开发集评估中变量预筛选及最终重新优化的潜在乐观偏倚，拟补充共同外层划分下的流程验证，并在修订中明确历史分析与新增敏感性分析的区别。”')
 doc.add_paragraph('只有实际运行并核验后，才能改成已完成时态并填入结果。不能写“原研究所有结局相关步骤均已严格嵌套”，也不能称本次重复使用的外部队列为全新未触碰验证集。')
 path=out/'实验方法核查_中文版.docx';doc.save(path)
 write_json(out/'method_audit.json',{'time':now(),'verdict':'revision_required','supported_scope':'six-model fixed-fusion FT sensitivity, 1050 development / 309 reused external',
  'original_pipeline_fully_nested_confirmed':False,'catboost_reoptimization_issue':'global search followed by CV of selected estimator',
  'not_covered':['Frozen three models','1059 CatBoost full-pipeline validation','shape ablation','final-model clinical and acquisition analyses'],
  'sources':[{'path':str(PROJECT/x),'sha256':sha(PROJECT/x)} for x in ['JMRI-26-1970_审稿意见.docx','JMRI-26-1970-ME-EIC-edits.docx']],
  'rows':df.to_dict('records')})
 print(path)

if __name__=='__main__':main()
