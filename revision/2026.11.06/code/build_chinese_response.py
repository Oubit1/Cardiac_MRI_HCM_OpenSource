"""Generate the Chinese mViT reviewer response from verified completed results."""
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
M=pd.read_csv(ROOT/'tables/Performance.csv')
P=pd.read_csv(ROOT/'tables/Paired_comparisons.csv')
F=pd.read_csv(ROOT/'tables/Fold_checkpoint_selection.csv')
S=pd.read_csv(ROOT/'tables/Acquisition_subgroups.csv')
ARMS=['baseline20_full','extended60_full','roi_matched_full']
NAMES={'baseline20_full':'整图：20轮选模窗口','extended60_full':'整图：60轮选模窗口',
       'roi_matched_full':'ROI：匹配整图训练时长','CMR_fixed':'CMR（固定融合分类器）',
       'Rscore_CMR_fixed':'Rscore＋CMR（固定融合分类器）',
       'mViT_CMR_fixed':'mViT＋CMR（固定融合分类器）',
       'mViT_Rscore_CMR_fixed':'mViT＋Rscore＋CMR（固定融合分类器）'}


def row(arm,dataset='External test'):
    return M[(M.Model==arm)&(M.Dataset==dataset)].iloc[0]


def ci(x,key='AUC'):
    return f'{x[key]:.3f}（{x[key+"_CI_low"]:.3f}～{x[key+"_CI_high"]:.3f}）'


def main():
    assert json.loads((ROOT/'FINAL_COMPLETE.json').read_text())['completed']
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=sec.bottom_margin=Cm(2)
    sec.left_margin=sec.right_margin=Cm(2)
    for name in ['Normal','Title','Heading 1','Heading 2','Caption']:
        style=doc.styles[name];style.font.name='Times New Roman'
        style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体' if name=='Normal' else '黑体')
    doc.styles['Normal'].font.size=Pt(10.5)
    doc.styles['Normal'].paragraph_format.line_spacing=1.2
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    footer=sec.footer.paragraphs[0];footer.alignment=1
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    plain=[]
    def paragraph(text):
        p=doc.add_paragraph(text);plain.append(text)
        if text=='回复：':p.paragraph_format.keep_with_next=True
        return p
    def heading(text,level=1):
        doc.add_heading(text,level=level);plain.extend(['',text])
    def section(title,comment,paragraphs,evidence):
        heading(title)
        paragraph('审稿意见要点（概述）：'+comment)
        paragraph('回复：')
        for text in paragraphs:paragraph(text)
        paragraph('对应材料：'+evidence+'。修订稿的具体页码、行号待整合后补充。')
    def table(title,headers,rows,note=''):
        paragraph(title).paragraph_format.keep_with_next=True
        t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid'
        for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in rows:
            for cell,text in zip(t.add_row().cells,values):cell.text=str(text)
        for rr in t.rows:
            rr._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell in rr.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size=Pt(9)
                        run.font.name='Times New Roman'
                        run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
        plain.append('\t'.join(headers))
        plain.extend('\t'.join(map(str,x)) for x in rows)
        if note:paragraph('注：'+note)

    doc.add_heading('审稿意见回复（中文版）',0)
    paragraph('稿件编号：JMRI-26-1970｜范围：mViT相关补充实验与说明｜结果版本：2026年11月7日')
    paragraph('本文件根据已完成的核心补充实验及英文回复整理。以下审稿意见为要点概述，并非逐字引述；未修改的论文内容不写成已完成修订。正文可用于后续修订整合，文末“作者内部核对说明”供作者核查，不建议直接原样作为审稿回复提交。配图沿用英文投稿图件，附中文图注。本回复的表1—4仅用于组织说明，不代表稿件最终表号；“原投稿Table 4”另指已核实的投稿比较表。')
    paragraph('感谢编辑和审稿人提出的具体建议。我们补充了五折训练时长敏感性分析、训练时长匹配的心肌ROI对照、外部判别与校准评估，以及固定分类器下的融合敏感性分析。我们将区分模型表达能力、训练过程、输入监督信息和实际泛化表现，并据此收敛结论。')

    section('一、审稿人1：仅训练20轮是否充分；请提供逐轮训练和验证曲线',
        '20轮训练上限可能限制微调效果，需要更长训练及逐轮性能曲线来判断。',[
        '我们同意，仅设置20轮上限不足以证明模型已充分优化。新增实验在五个外层训练折内各自保留10%的患者作为内部验证集，使用固定编码器学习率1×10⁻⁵完成60轮训练，不提前停止；分别在前20轮和全部60轮中，依据内部验证AUC选择检查点。之后从同一预训练初始化出发，在完整外层训练集按选定轮数重训，再预测未参与训练和选模的外层留出患者。',
        '每折外层训练集为840例，外层留出集为210例；内部训练与验证分别为756例和84例。五折共获得1,050例患者的折外预测。逐轮记录训练与内部验证的AUC和二元交叉熵，训练评估关闭数据增强。训练曲线用于分析优化过程，不能替代外层留出评估。',
        '扩展至60轮后，五折原始最佳检查点分别位于第10、13、2、31、6轮。第4折出现了20轮之后的内部验证改善，因此我们不将结果概括为“20轮在所有情况下均已充分”。另外四折的原始最佳检查点位于前20轮。',
        '20轮窗口与60轮窗口方案的开发折外AUC分别为0.754和0.758，配对差值为＋0.005（95% CI：−0.028～＋0.033；Holm校正P＝0.762）。外部AUC分别为0.808和0.797，差值为−0.011（95% CI：−0.030～＋0.007；Holm校正P＝0.484）。本实验没有显示延长选模窗口带来明确的外部判别改善，但不能据此证明两方案等效或模型已达到全局最优。',
        '这里的“20轮/60轮”指内部检查点选择窗口，而不是最终模型必须训练的轮数。重训采用预定的至少3轮规则，使其包含编码器解冻后的训练；因此原始最佳检查点为第2轮时，实际重训为3轮。原始检查点和实际重训轮数分别报告，不将这一规则解释为性能最优。'],
        '图S1、S5、S6；Excel中的Epoch_history、Fold_checkpoint_selection和Paired_comparisons')
    table('表1　各折检查点选择与实际重训轮数',
          ['外层折','20轮窗口最佳轮','60轮窗口最佳轮','20轮方案重训','60轮方案重训','ROI重训'],
          [[int(x.Fold),int(x.Checkpoint_epoch_20),int(x.Checkpoint_epoch_60),int(x.Refit_epoch_20),int(x.Refit_epoch_60),int(x.ROI_refit_epoch)] for x in F.itertuples()],
          'ROI使用对应整图60轮窗口方案的重训时长。配置完全相同的重训复用同一检查点，不能作为独立重复实验。')
    table('表2　新增图像模型的判别性能', ['方案','开发折外AUC（95% CI）','外部AUC（95% CI）'],
          [[NAMES[a],ci(row(a,'Development OOF')),ci(row(a))] for a in ARMS],
          '开发集1,050例，外部集309例。区间通过2,000次按结局分层的患者bootstrap获得，反映固定预测下的患者抽样不确定性，并不包含完整重新训练的不确定性。')

    section('二、审稿人1：AdamW参数与微调策略的依据',
        '请解释优化器超参数及微调策略的选择。',[
        '我们采用AdamW，将权重衰减与梯度更新中的自适应缩放分开处理。编码器学习率设为1×10⁻⁵，注意力聚合和分类头为1×10⁻⁴：较小的编码器学习率用于限制对预训练表示的剧烈更新，而新初始化的任务模块需要较快适应当前任务。该解释属于设计动机，不代表这些参数已被证明最优。',
        '前2轮冻结编码器，之后全部解冻；dropout为0.25，权重衰减为1×10⁻⁴，梯度范数裁剪上限为1.0。每次输入一位患者的切片集合，累积4位患者的梯度后更新参数；最多8张切片的编码器微批大小与患者批大小是不同概念。采用float16混合精度和GradScaler。',
        '本次已完成的核心实验固定上述参数，主要检验训练时长和输入定位。尚未完成学习率网格与额外随机种子实验，因此不会声称已完成全面超参数搜索或证实随机种子稳健性。'],
        'Excel中的Protocol、Training_runs和Execution_sessions')
    table('表3　核心训练设置',['项目','实际设置'],[
        ['AdamW β₁ / β₂ / ε','0.9 / 0.999 / 1×10⁻⁸'],['编码器 / 任务头学习率','1×10⁻⁵ / 1×10⁻⁴'],
        ['权重衰减 / dropout','1×10⁻⁴ / 0.25'],['冻结与解冻','前2轮冻结编码器，第3轮开始全部解冻'],
        ['梯度累积 / 梯度裁剪','4位患者 / 范数1.0'],['每位患者输入','最多8张切片，每张16帧，224×224像素'],
        ['选择规则','内部验证AUC改善超过1×10⁻⁴时更新，未达到改善条件时保留更早检查点'],
        ['学习率调度','固定学习率；未使用余弦调度或学习率线性预热'],
        ['外部阈值','由开发折外预测确定，要求敏感度至少0.95时取最高特异度；外部不重调']])

    section('三、审稿人1：外部AUC由哪个模型产生',
        '不同交叉验证折得到不同模型，外部队列使用哪个模型？',[
        '对于本次新增实验，外部测试使用在全部1,050例开发患者上重新训练的模型，而不是根据外部AUC挑选某一折的模型。重训从相同预训练权重开始，学习率固定；训练轮数取五折实际选定重训轮数的中位数，并在任何新增外部推断前锁定。',
        '最终20轮窗口方案重训6轮，60轮窗口方案重训10轮，ROI方案重训10轮。三个最终模型随后在同一309例外部患者上评价。原投稿模型的来源与本次新增模型分开核查，不能用新增实验配置倒推原投稿配置。'],
        'plans/final_locked_configs.json及Excel中的Training_runs')

    section('四、审稿人1：通用逼近能力、完整cine输入与增量价值',
        'mViT使用完整cine却未超过单时相Rscore，是否说明训练不足；不能仅用信息重叠解释结果。',[
        '我们同意，原先仅用“信息部分重叠”解释结果不够充分。通用逼近相关结论讨论的是特定假设下的函数表示能力，并不保证一个固定mViT架构在有限样本、有限训练预算和既定输入预处理下能够学到手工放射组学映射，或达到更高的外部AUC。Rscore使用专家心肌分割，还获得了整图mViT没有被显式提供的定位信息。',
        '“mViT＋CMR低于Rscore＋CMR”与“向Rscore＋CMR加入mViT未获得进一步改善”是两个不同问题，均不能直接证明生物学信息冗余。补充的训练时长和ROI实验没有显示明确的外部判别改善，但仍不能排除其他优化方法、学习率或更可靠定位监督带来的收益。',
        '另需澄清模型名称：核实后的原投稿Table 4中，mViT-Frozen＋CMR外部AUC为0.786，mViT-FT＋CMR为0.821。因此约0.82的结果对应微调组合，而非冻结组合。',
        '为进一步评估当前表示的增量价值，我们以预先固定的弹性网络逻辑回归设置进行外部融合敏感性分析（C＝0.1，l1_ratio＝0.5）。CMR与mViT＋CMR的外部AUC分别为0.815和0.811；Rscore＋CMR与加入mViT后的外部AUC分别为0.872和0.827。后一差值为−0.045（95% CI：−0.076～−0.016，未校正DeLong P＝0.00265）。这说明当前表示与固定融合方案没有带来增量收益，不能推广为所有深度学习方法均不如放射组学。该分析仅作探索性外部结果报告，不提出新的融合折外验证独立性主张。',
        '我们还对已有完整cine与重复首帧预测按当前标签重新评价。外部AUC分别为0.802和0.739，配对差值为＋0.063（95% CI：＋0.022～＋0.104，未校正DeLong P＝0.00166）。这提示既有实现中的时间信息有贡献，但不能证明训练已经最优。此为历史预测再分析，并未新训练单帧模型；第0帧尚未经独立核实为舒张末期，因此称为“重复首帧”，不直接写作“ED重复”。'],
        '图S6；Excel中的Submitted_Table4、Performance、Paired_comparisons及Temporal_archived系列工作表')

    cache=json.loads((ROOT/'audit/cache_summary.json').read_text())
    section('五、审稿人3第5条：心肌定位监督与比较公平性',
        '放射组学使用专家分割，而mViT输入包含心腔及心外结构，建议增加心肌ROI模型或更充分讨论局限。',[
        '我们认同这两类方法的输入监督并不对等。新增探索性ROI实验使用cine心肌掩膜在整个心动周期中的并集，为每张切片形成固定裁剪框，并在每侧增加20%边距；同一切片各时相使用相同裁剪框，不使用LGE图像或LGE衍生区域。',
        'ROI与对应整图模型采用相同训练患者、初始化种子、学习率和训练时长；训练时长由整图内部验证选择。因此该实验评估的是训练预算匹配条件下的定位敏感性，而不是单独优化后的最佳ROI模型。',
        f'现有cine二值掩膜的生成来源尚未完全核实，不能称作专家手工分割真值。掩膜不可用或未通过质控时按既定规则回退整图，未因掩膜问题排除患者。共{cache["slices"]:,}张入选切片中{cache["roi_fallback_slices"]:,}张回退整图（{100*cache["roi_fallback_slices"]/cache["slices"]:.2f}%）；这是切片层面比例，不是患者比例。',
        '整图60轮窗口方案与ROI方案的外部AUC分别为0.797和0.793，ROI减去整图的差值为−0.003（95% CI：−0.034～＋0.026；Holm校正P＝0.815）。当前结果未显示明确的ROI收益，但不能证明定位信息无用，也不能认为输入监督公平性问题已被完全解决。我们拟在局限中明确保留这一点。'],
        '图S3—S6；Excel中的ROI_QC、Performance和Paired_comparisons')

    section('六、审稿人1：场强、平面内分辨率及其与LGE和性能的关系',
        '请报告场强分布、与LGE状态的关联，以及采集参数对模型性能的影响。',[
        '在本次匹配分析队列中，开发集1,050例均为3.0 T；外部集1.5 T和3.0 T分别为157例和152例，其中LGE阳性分别为104例和108例。外部场强与LGE状态的Fisher精确检验P＝0.392；开发集只有一个场强水平，不能在其内部估计场强与结局的关联。',
        '平面内分辨率以两个方向像素间距的几何平均值表示。开发集LGE阴性、阳性组的中位值分别为0.938和0.994 mm，Mann–Whitney检验P＝5.93×10⁻⁸；外部两组中位值均为1.231 mm，P＝0.118。这些关联可能混合了中心、设备和协议差异，不能作因果解释。',
        '外部性能按场强及开发集分辨率中位数0.9375 mm分层；阈值仍使用开发折外预测确定的值。以整图60轮窗口方案为例，1.5 T和3.0 T亚组AUC分别为0.827（95% CI：0.748～0.897）和0.790（0.698～0.872）。分辨率不高于与高于该像素间距界值的亚组AUC分别为0.771（0.564～0.950）和0.803（0.741～0.861）。',
        '较小分辨率亚组仅24例，估计不精确。我们报告分层估计及区间，不将区间重叠解释为亚组等效，也不把未经专门交互检验的数值差异写成已证实的场强或分辨率效应。当前分层不能分离中心、设备厂商、场强和协议影响。'],
        '图S7；Excel中的Acquisition_LGE_association和Acquisition_subgroups')

    section('七、审稿人3第8条及编辑：校准与锁定阈值的外部表现',
        '请在校准图之外提供校准截距和斜率的95%置信区间，并谨慎解释临床用途。',[
        '我们补充了外部校准图、Brier分数及联合估计的校准截距和斜率，并提供患者bootstrap置信区间。这里的截距来自同时估计截距与斜率的校准模型，不是将斜率固定为1的偏置项校准。',
        '60轮窗口方案在开发折外预测上锁定的阈值为0.006691。应用到外部全开发集重训模型后，309例全部预测为阳性：TP＝212、FP＝97、TN＝0、FN＝0，敏感度为1.000、特异度为0.000。由于没有预测阴性患者，NPV无定义。该操作点没有提供阴性分流人群，不能仅凭AUC支持其用于筛选可省略增强检查的患者。',
        '这一表现提示判别性能与阈值可迁移性需要分别评价；汇总折外概率选定的阈值未能有效迁移到该全开发集重训模型，是与结果一致的解释，但尚不能视为已验证的机制。我们没有为了改善结果而使用外部标签重选阈值，也没有把探索性外部结果写成临床可用性的证明。'],
        '图S4；Excel中的Performance、Calibration_bins及逐例预测')
    table('表4　新增模型的外部校准与阈值表现',
          ['方案','校准截距（95% CI）','校准斜率（95% CI）','敏感度','特异度'],
          [[NAMES[a],ci(row(a),'Calibration_intercept_joint'),ci(row(a),'Calibration_slope_joint'),f'{row(a).Sensitivity:.3f}',f'{row(a).Specificity:.3f}'] for a in ARMS],
          '开发折外阈值固定后应用于外部。校准图中的分箱误差条为观察事件比例的Wilson区间，与校准系数bootstrap区间不同。')

    section('八、审稿人1：开发AUC低于外部AUC如何解释',
        '原投稿mViT-FT开发AUC为0.793、外部为0.802，这是否提示训练不足？',[
        '这两个数值来自不同患者及不同模型训练样本量：开发结果为折外评估，外部结果来自对应最终模型。抽样变异、病例构成和训练样本量差异均可能导致外部AUC略高。该数值关系本身既不能证明更强的可迁移性，也不能证明训练不足。',
        '我们通过新增训练曲线和训练时长对照直接检查优化问题，而不是用两个AUC的大小关系替代证据。新增折外AUC与历史投稿数值也不能当作同一流程的重复测量：新增实验明确采用内部选模、完整外层训练集重训和外层留出预测，历史流程的局限仍需单独说明。'],
        '图S1；Excel中的Submitted_Table4、Performance及Protocol')

    section('九、编辑：外部队列是否重复使用；所有使用结局信息的步骤是否嵌套',
        '需说明外部结果已知之后的模型决策，以及CMR、Rscore、mViT适应与融合的完整验证独立性。',[
        '本次使用的是此前已经评价过的同一外部队列，因此新增结果应称作修订阶段的敏感性评价，不能称作首次使用、完全未触及的确认性外部验证。本次新增模型的实际检查点、最终重训配置与阈值均依据开发数据锁定，再进行新增外部推断；但这不能消除该队列曾被查看的事实。',
        '本次新增纯图像分析中，外层留出患者不参与监督式mViT适应或检查点选择。每个外层训练集再划分内部验证集选择轮数，然后在完整外层训练集重训。核心方案的学习率固定，未进行新的学习率网格选择。',
        '完整下游融合流程的独立性不能仅凭“使用了折外特征”来保证。作者确认的CMR脚本在完整开发集进行变量选择；预先生成的全局折外Rscore也不自动使后续融合交叉验证完全独立。新版Rscore脚本的嵌套结构不能追溯性证明历史CMR、mViT表示与融合组成的整个流程均已嵌套。',
        '因此，新增表示融合只报告预先固定分类器的探索性外部结果，不报告新的融合折外性能，不将全开发集监督表示当作独立折外特征。历史外部结果知晓后是否还存在其他调参决策，应由作者结合实际记录补充时间线；仅存在相关工具代码，不能证明该工具用于投稿模型，也不能证明从未使用。'],
        'Excel中的Protocol、Training_runs和Deferred_experiments，以及final_locked_configs.json')

    heading('作者内部核对说明（不建议直接作为审稿回复原样提交）')
    paragraph('1. 原投稿Table 4逐例预测可重算得到开发AUC＝0.792735、外部AUC＝0.801692。旧归档具有相同图像预测概率，但6例外部结局标签不同，可解释旧记录的0.836632。采用的当前标签与现有临床和放射组学表一致；标签版本差异不能当作模型改进。')
    paragraph('2. 候选历史检查点在309例外部数据上得到相同AUC，但与投稿逐例概率的最大差值为0.00146484，超过事先设定的0.001容差。因此只能写作候选检查点近似复现，不能声称逐例精确匹配或完全确认了投稿权重身份。')
    paragraph('3. 现有方法实现是按均匀位置取整选择16帧，不是线性时间插值；分类头为LayerNorm后的两层MLP。首帧是否为ED仍需核实，cine掩膜来源和原投稿外部调参时间线仍需作者确认。')
    paragraph('4. 已完成的是约定的核心补实验。学习率网格、独立ROI长训练调参及额外随机种子未执行，不应写入“已完成实验”。加速实现与第1折第51轮开始的执行切换已记录；混合精度数值接近不代表逐位一致或已证明泛化等效。')
    paragraph('5. 上文提供的是可用于整合的回复内容。尚未在编辑提供的EIC标注稿中完成全部正文修改和页码定位，提交前应核对正文、表格、图注与回复的一致性。本文件仅覆盖mViT相关内容，不代替其余临床、Rscore消融或伦理信息的回复。')

    captions={
        'S0':'原投稿模型在当前标签下的外部ROC与描述性校准图，属于历史预测复核，并非新增训练结果。',
        'S1':'五折逐轮训练和内部验证曲线。训练评估关闭增强；虚线标记第20轮，灰色区表示冻结编码器的前2轮。内部验证用于选模，不代表外层留出性能。',
        'S2':'固定编码器学习率下各折最佳内部验证AUC。本核心方案没有完成学习率网格搜索。',
        'S3':'新增模型ROC曲线。OOF表示汇总全部五折的外层留出预测；external表示配置锁定后的外部评价。',
        'S4':'描述性校准图，按预测概率形成至多10个等频分箱；误差条为各箱观察事件比例的95% Wilson区间，不是拟合校准函数的置信带。',
        'S5':'AUC及2,000次按结局分层患者bootstrap得到的95%区间。单个区间的重叠不能替代配对检验。',
        'S6':'配对AUC差值为模型B减模型A；横线为95% bootstrap区间，零线为无观察差异。主要图像比较的Holm校正P与探索性融合的未校正P在Excel中分别报告。',
        'S7':'外部探索性采集参数亚组AUC及95%区间。分辨率以平面内像素间距几何均值表示，使用开发中位数分组。图中列出样本量和事件数；中心、设备、场强和协议影响未被独立分离。'}
    for path in sorted((ROOT/'figures').glob('Figure_*.png')):
        doc.add_page_break();heading('配图：'+path.stem,2)
        doc.add_picture(str(path),width=Cm(17))
        paragraph(captions[path.stem.split('_')[1]])
    heading('数据与文件索引')
    paragraph('完整数值、逐例预测、逐轮历史、混淆矩阵、阈值、校准系数、置信区间与来源核查见 tables/mViT_revision_results.xlsx。英文回复为 response/mViT_response.docx。全部结果位于2026.11.06目录；投稿配图同时提供PNG、PDF及300 dpi TIFF。')
    output=ROOT/'response/mViT_response_中文版.docx'
    doc.save(output)
    (ROOT/'response/mViT_response_中文版.txt').write_text('\n\n'.join(plain),encoding='utf-8')
    check=Document(output)
    assert len(check.inline_shapes)==13 and len(check.tables)==4
    assert '特异度为0.000' in '\n'.join(p.text for p in check.paragraphs)
    print('Chinese response generated:',output)


if __name__=='__main__':main()
