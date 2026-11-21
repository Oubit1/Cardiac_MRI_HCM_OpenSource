"""Reports only completed predictions; pending runs never receive invented results."""
from settings import *
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import metrics_ci,paired,holm
from sklearn.metrics import roc_curve

def oof_predictions():
    frames=[]
    for fold in range(1,6):
        directory=ROOT/f'runs/fold_{fold}'
        if not (directory/'fusion_complete.json').exists():raise RuntimeError('Five completed outer folds required')
        frame=pd.read_csv(directory/'predictions.csv');frame['fold']=fold
        spec=json.loads((directory/'split.json').read_text())
        assert frame.ID.tolist()==spec['outer_test']
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    labels=pd.read_csv(ROOT/'inputs/evaluation_labels.csv').query("cohort=='development'")
    assert len(result)==1050 and not result.ID.duplicated().any() and set(result.ID)==set(labels.ID)
    return result.merge(labels[['ID','event']],on='ID',validate='one_to_one')

def lock_thresholds():
    data=oof_predictions();values={}
    for name in MODELS:
        fpr,tpr,cut=roc_curve(data.event,data[name],drop_intermediate=False)
        eligible=np.flatnonzero((tpr>=.95)&np.isfinite(cut))
        best=eligible[np.isclose(fpr[eligible],fpr[eligible].min())]
        # Match historical CatBoost tie convention explicitly; it is a new model's threshold.
        values[name]=float(cut[best].max())
    signature=__import__('hashlib').sha256(data.to_csv(index=False).encode()).hexdigest()
    path=ROOT/'plans/locked_thresholds.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['oof_sha256']!=signature or old['thresholds']!=values:raise RuntimeError('Locked OOF thresholds changed')
        return old
    if (ROOT/'runs/final/predictions.csv').exists():raise RuntimeError('Refuse retrospective threshold lock after final predictions')
    result={'time':now(),'oof_sha256':signature,'thresholds':values,'rule':'sensitivity >=0.95, maximum specificity, then highest finite threshold',
            'external_cohort_previously_examined':True}
    write_json(path,result);return result

def main():
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    for name in ['tables','figures','response']:(ROOT/name).mkdir(exist_ok=True)
    completed=[i for i in range(1,6) if (ROOT/f'runs/fold_{i}/fusion_complete.json').exists()]
    doc=Document();style=doc.styles['Normal'];style.font.size=Pt(10);style.font.name='Calibri'
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    doc.add_heading('共同五折验证：九模型匹配敏感性分析',0)
    doc.add_paragraph('范围：1050例开发集、309例重复使用的外部队列；固定弹性网融合参数。包含Frozen与FT及其组合；1059例CatBoost由独立分支生成报告。')
    doc.add_paragraph('历史外部结果已被查看，本次是修订阶段敏感性分析。训练集mViT特征为对应外层模型的拟合内表示；外层测试病例未参与该折监督训练。')
    if len(completed)<5:
        doc.add_paragraph(f'状态：已完成融合的外层折数 {len(completed)}/5；尚无可报告的完整OOF性能。')
        doc.save(ROOT/'response/nested_validation_response_中文版.docx')
        write_json(ROOT/'audit/report_validation.json',{'completed':False,'folds':completed,'reason':'awaiting all five folds'})
        return
    datasets={'development_OOF':oof_predictions()}
    thresholds=lock_thresholds()['thresholds']
    if (ROOT/'runs/final/fusion_complete.json').exists():
        ext=pd.read_csv(ROOT/'runs/final/predictions.csv')
        labels=pd.read_csv(ROOT/'inputs/evaluation_labels.csv').query("cohort=='external_test'")
        assert len(ext)==309 and not ext.ID.duplicated().any() and set(ext.ID)==set(labels.ID)
        datasets['external_reused']=ext.merge(labels[['ID','event']],on='ID',validate='one_to_one')
    policy=json.loads((ROOT/'plans/protocol.json').read_text())
    signature=__import__('hashlib').sha256((''.join(x.to_csv(index=False) for x in datasets.values())+json.dumps(thresholds)+sha(__file__)+sha(ROOT/'plans/protocol.json')).encode()).hexdigest()
    cache=ROOT/'audit/report_statistics_cache.json'
    if cache.exists() and json.loads(cache.read_text()).get('signature')==signature:
        saved=json.loads(cache.read_text());metrics=saved['metrics'];comparisons=saved['comparisons']
    else:
        metrics=[];comparisons=[]
        for cohort,data in datasets.items():
            for name in MODELS:
                cut=None if cohort=='development_OOF' else thresholds[name]
                metrics.append({'cohort':cohort,'model':name,**metrics_ci(data.event,data[name],cut,rounds=policy['statistics']['patient_stratified_bootstrap'])})
            pair_rows=[]
            for a,b in policy['primary_comparisons']:
                pair_rows.append({'cohort':cohort,'A':a,'B':b,**paired(data.event,data[a],data[b],rounds=policy['statistics']['patient_stratified_bootstrap'])})
            for row,adj in zip(pair_rows,holm([x['DeLong_P'] for x in pair_rows])):row['Holm_P']=float(adj)
            comparisons.extend(pair_rows)
        write_json(cache,{'signature':signature,'metrics':metrics,'comparisons':comparisons})
    metrics=pd.DataFrame(metrics);comparisons=pd.DataFrame(comparisons)
    logs=[];selections=[];grids=[];coefficients=[]
    for folder in [ROOT/f'runs/fold_{i}' for i in range(1,6)]+[ROOT/'runs/final']:
        for file in ['r_complete.json','features_complete.json','fusion_complete.json']:
            if (folder/file).exists():
                item=json.loads((folder/file).read_text());logs.append({'run':folder.name,'stage':file,'seconds':item.get('seconds'),'metadata':json.dumps(item,ensure_ascii=False)[:30000]})
        for file in folder.glob('*/selection.json'):
            item=json.loads(file.read_text());selections.append({'run':folder.name,'stage':file.parent.name,**{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in item.items()}})
        for filename,collection in [('tuning_grid.csv',grids),('coefficients.csv',coefficients)]:
            for file in folder.glob('*/'+filename):
                frame=pd.read_csv(file);frame.insert(0,'stage',file.parent.name);frame.insert(0,'run',folder.name);collection.append(frame)
    with pd.ExcelWriter(ROOT/'tables/nested_validation_results.xlsx',engine='openpyxl') as writer:
        sheets={'Metrics':metrics,'Paired_comparisons':comparisons,'Locked_thresholds':pd.DataFrame(thresholds.items(),columns=['model','threshold']),
                'Selection':pd.DataFrame(selections),'Execution':pd.DataFrame(logs),
                'Protocol':pd.DataFrame([{'key':k,'value':json.dumps(v,ensure_ascii=False)[:30000]} for k,v in policy.items() if k!='folds']),
                **datasets}
        if grids:sheets['All_tuning']=pd.concat(grids,ignore_index=True)
        if coefficients:sheets['All_coefficients']=pd.concat(coefficients,ignore_index=True)
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False);ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            from openpyxl.styles import Font,PatternFill
            for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='17365D')
            for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=min(45,max(15,len(str(col[0].value))+3))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    for cohort,data in datasets.items():
        fig,axes=plt.subplots(1,2,figsize=(10,4.4))
        for name in MODELS:
            row=metrics.query('cohort==@cohort and model==@name').iloc[0]
            fpr,tpr,_=roc_curve(data.event,data[name]);axes[0].plot(fpr,tpr,label=f'{name}: {row.AUC:.3f}')
            observed,mean=calibration_curve(data.event,data[name],n_bins=8,strategy='quantile');axes[1].plot(mean,observed,'o-',ms=3,label=name)
        for ax in axes:ax.plot([0,1],[0,1],'k--',lw=.8);ax.set_xlim(0,1);ax.set_ylim(0,1)
        axes[0].set(xlabel='1 - specificity',ylabel='Sensitivity',title='Discrimination');axes[0].legend(fontsize=7,loc='lower right')
        axes[1].set(xlabel='Mean estimated probability',ylabel='Observed LGE fraction',title='Calibration');axes[1].legend(fontsize=7)
        fig.suptitle('Development OOF' if cohort=='development_OOF' else 'Previously evaluated external cohort');fig.tight_layout()
        for extension in ['png','pdf','tiff']:fig.savefig(ROOT/f'figures/{cohort}.{extension}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    doc.add_heading('结果',1)
    table=doc.add_table(rows=1,cols=4);table.style='Light Shading Accent 1'
    for cell,text in zip(table.rows[0].cells,['队列','模型','AUC（95% CI）','Brier']):cell.text=text
    for _,row in metrics.iterrows():
        for cell,text in zip(table.add_row().cells,[row.cohort,row.model,f'{row.AUC:.3f} ({row.AUC_CI_low:.3f}, {row.AUC_CI_high:.3f})',f'{row.Brier:.3f}']):cell.text=text
    doc.add_paragraph('开发OOF仅报告判别、Brier和校准；阈值在这批OOF上选择，故不把相应OOF敏感度/特异度当作独立验证。外部操作指标及联合估计的校准截距、斜率和置信区间详见Excel。bootstrap与DeLong基于现有逐例预测，未重新训练模型，不涵盖完整模型选择的不确定性。')
    for _,row in comparisons.iterrows():doc.add_paragraph(f'{row.cohort}：{row.B} 相对 {row.A}，ΔAUC={row.Delta_AUC_B_minus_A:.3f}，95% CI [{row.Delta_CI_low:.3f}, {row.Delta_CI_high:.3f}]，Holm P={row.Holm_P:.4g}。')
    doc.add_paragraph('以上是新增分析的结果，不能追溯证明历史流程没有信息泄漏；也不支持仅凭阴性结果省略对比剂。结论应限定于本次数据、预处理和固定参数。')
    for cohort in datasets:doc.add_picture(str(ROOT/f'figures/{cohort}.png'),width=__import__('docx').shared.Inches(6.2))
    doc.save(ROOT/'response/nested_validation_response_中文版.docx')
    write_json(ROOT/'audit/report_validation.json',{'completed':'external_reused' in datasets,'cohorts':{k:len(v) for k,v in datasets.items()},'signature':signature,'scope':'nine-model matched sensitivity'})

if __name__=='__main__':main()
