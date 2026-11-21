"""Wait for the running suite, validate outputs and package the completed revision experiments."""
from settings import *
import time,subprocess,zipfile,shutil
import pandas as pd,numpy as np
import psutil,fcntl

def validate():
    from settings import MODELS
    from tabular_suite import OUTPUTS,BASE
    checks=[]
    for fold in range(1,6):
        d=ROOT/f'runs/fold_{fold}';split=json.loads((d/'split.json').read_text())
        train=set(split['outer_train']);held=set(split['outer_test'])
        assert train.isdisjoint(held)
        pred=pd.read_csv(d/'predictions.csv');assert set(pred.ID)==held and not pred.ID.duplicated().any()
        assert np.isfinite(pred[MODELS]).all().all()
        for path in d.glob('*/selection.json'):
            ids=set(json.loads(path.read_text())['training_ids']);assert ids<=train and ids.isdisjoint(held)
        checks.append({'branch':'matched','fold':fold,'n':len(pred),'passed':True})
    for variant in ['all_features','no_shape']:
        root=BASE/variant;folds=json.loads((root/'folds.json').read_text())
        if variant=='no_shape':assert not any('_shape_' in x.lower() or '_shape2d_' in x.lower() for x in json.loads((root/'protocol.json').read_text())['rscore_features'])
        for fold in folds:
            number=fold['split'];train=set(fold['training_ids']);held=set(fold['heldout_ids'])
            d=root/f'fold_{number}';pred=pd.read_csv(d/'outer/predictions.csv')
            assert set(pred.ID)==held and not pred.ID.duplicated().any() and np.isfinite(pred[OUTPUTS]).all().all()
            for path in d.rglob('selection.json'):
                ids=set(json.loads(path.read_text())['training_ids']);assert ids<=train and ids.isdisjoint(held)
            for inner in d.glob('inner_*'):
                inner_held=set(pd.read_csv(inner/'prediction_features.csv').ID)
                for path in inner.rglob('selection.json'):assert set(json.loads(path.read_text())['training_ids']).isdisjoint(inner_held)
            checks.append({'branch':variant,'fold':number,'n':len(pred),'passed':True})
        ext=pd.read_csv(root/'final/predictions.csv');assert len(ext)==309 and not ext.ID.duplicated().any()
    write_json(ROOT/'audit/final_delivery_validation.json',{'passed':True,'time':now(),'checks':checks})

def package():
    from docx import Document
    from docx.shared import Inches
    from docx.oxml.ns import qn
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    doc=Document();doc.styles['Normal'].font.name='Calibri';doc.styles['Normal'].element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    doc.add_heading('JMRI修订：验证实验中文回复',0)
    doc.add_paragraph('本文件根据本轮实际完成的验证结果自动生成。新增实验采用一次外层五折；不将其表述为原投稿五次重复的精确复现。外部队列此前已经使用，本轮结果作为修订敏感性证据。')
    doc.add_heading('编辑：结局相关步骤是否嵌套在相应评价折内？',1)
    doc.add_paragraph('感谢编辑指出这一关键问题。我们补充了共同外层划分下的流程验证。在每个外层训练集中重新完成传统MRI变量筛选、Rscore构建及下游拟合；在需要进行分类器参数与类型选择的分析中，相应上游步骤也在每个内层训练分区重新完成。用于融合训练的Rscore在其所属训练分区内交叉拟合，外层测试患者不参与这些拟合。')
    doc.add_paragraph('1050例匹配分析复用与外层划分严格对应的mViT权重，分别提取训练和测试表示；训练患者表示为拟合内表示，并未称为OOF。下游弹性网参数固定，以避免用外层监督编码器生成的内层验证表示进行调参。冻结编码器未使用本研究结局进行适配。')
    doc.add_heading('Reviewer 3：CMR筛选及CatBoost重新优化',1)
    doc.add_paragraph('我们补充了1059例开发集的全流程外层评价。五种分类器各采用四组预先固定的候选参数；参数选择和分类器选择在每个外层训练集的五折内层评价中进行。另报告固定弹性网参照。最终模型在开发集上确定后，使用开发OOF确定的阈值评价外部队列。该新增分析用于检验结论稳健性，不追溯宣称原先的全开发集预筛选及参数搜索不存在乐观偏倚。')
    doc.add_heading('Reviewer 3：形状特征是否驱动Rscore优势？',1)
    doc.add_paragraph('我们在所有监督筛选之前去掉13个形状特征，使用其余474个通过ICC筛选的一阶和纹理候选特征，重新执行完整的评分构建和嵌套验证，并检验其超出传统MRI指标的增量价值。')
    files=[('九模型匹配分析',ROOT/'tables/nested_validation_results.xlsx'),('1059例全流程及无形状分析',ROOT/'tabular_full/reports/full_validation_results.xlsx')]
    for title,file in files:
        data=pd.read_excel(file,sheet_name='Metrics')
        doc.add_heading(title,1)
        table=doc.add_table(rows=1,cols=4);table.style='Light Shading Accent 1'
        for cell,text in zip(table.rows[0].cells,['分析/队列','模型','AUC [95% CI]','Brier']):cell.text=text
        for _,row in data.iterrows():
            label=str(row.get('variant',''))+' / '+str(row['cohort'])
            vals=[label,row['model'],f"{row.AUC:.3f} [{row.AUC_CI_low:.3f}, {row.AUC_CI_high:.3f}]",f'{row.Brier:.3f}']
            for cell,text in zip(table.add_row().cells,vals):cell.text=str(text)
    pairs=pd.read_excel(ROOT/'tabular_full/reports/full_validation_results.xlsx',sheet_name='Comparisons')
    for _,r in pairs.query("cohort=='external'").iterrows():
        conclusion='观察到判别能力增量' if r.Delta_CI_low>0 else ('观察到判别能力下降' if r.Delta_CI_high<0 else '尚未显示明确的判别能力差异')
        doc.add_paragraph(f"{r.variant}：{r.B}相对{r.A}，ΔAUC={r.Delta_AUC_B_minus_A:.3f}（95% CI {r.Delta_CI_low:.3f}至{r.Delta_CI_high:.3f}；Holm校正P={r.Holm_P:.4g}），{conclusion}。")
    doc.add_paragraph('外部校准、阈值及操作指标详见完整Excel。我们将结论限定在实际观察结果，不据此宣称阴性患者可以安全省略对比剂。')
    for image in sorted((ROOT/'figures').glob('*.png'))+sorted((ROOT/'tabular_full/reports').glob('*.png')):
        doc.add_picture(str(image),width=Inches(6.2))
    response=delivery/'验证实验_response_中文版.docx';doc.save(response)
    with zipfile.ZipFile(delivery/'验证实验完整交付.zip','w',zipfile.ZIP_DEFLATED) as z:
        for directory in ['code','tables','figures','response','tabular_full/reports']:
            for path in (ROOT/directory).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:z.write(path,path.relative_to(ROOT))
        for path in [response,ROOT/'audit/final_delivery_validation.json',ROOT/'README.txt',ROOT/'run_experiments.sh']:z.write(path,path.relative_to(ROOT))
    write_json(delivery/'DELIVERY_COMPLETE.json',{'time':now(),'validated':True,'response':str(response),'archive':str(delivery/'验证实验完整交付.zip')})

def main():
    lock=(ROOT/'delivery.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pid=json.loads((ROOT/'suite_launcher.json').read_text())['pid']
    markers=[ROOT/'FINAL_COMPLETE.json',ROOT/'tabular_full/all_features/COMPLETE.json',ROOT/'tabular_full/no_shape/COMPLETE.json']
    def ready():
        state=json.loads((ROOT/'suite_status.json').read_text()).get('state')
        return all(p.exists() for p in markers) and state=='requested_stages_finished'
    while not ready():
        if not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:
            write_json(ROOT/'delivery_status.json',{'state':'experiment_controller_exited_before_completion','time':now(),'pid':pid})
            raise RuntimeError('Experiment controller exited; inspect experiments.log before restarting')
        write_json(ROOT/'delivery_status.json',{'state':'waiting_for_experiments','time':now(),'completed':[str(p.parent.relative_to(ROOT)) for p in markers if p.exists()]})
        time.sleep(30)
    subprocess.run([PYTHON,str(ROOT/'code/experiments.py'),'report'],check=True)
    validate();package()
    write_json(ROOT/'delivery_status.json',{'state':'complete','time':now()})

if __name__=='__main__':main()
