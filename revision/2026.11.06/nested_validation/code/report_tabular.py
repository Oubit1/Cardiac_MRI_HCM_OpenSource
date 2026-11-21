"""Tables, figures and Chinese results from completed full-cohort branches."""
from settings import *
from tabular_suite import BASE,OUTPUTS
import sys,numpy as np,pandas as pd
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import metrics_ci,paired,holm

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.calibration import calibration_curve
    from docx import Document
    from docx.shared import Inches
    from docx.oxml.ns import qn
    out=BASE/'reports';out.mkdir(exist_ok=True)
    metrics=[];pairs=[];predictions=[];tuning=[];selections=[];coefficients=[];timing=[]
    doc=Document();doc.styles['Normal'].font.name='Calibri';doc.styles['Normal'].element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    doc.add_heading('1059例全流程验证与无形状评分：结果',0)
    doc.add_paragraph('五折外层评价；每折内重新构建CMR、Rscore并完成五分类器的内层参数与类型选择。每类四组固定候选参数；新增分析采用一次五折，未冒称历史五次重复实验。Selected为每个外层训练集内选出的分类器策略。外部队列为此前已使用的309例。')
    for variant in ['all_features','no_shape']:
        root=BASE/variant
        if not (root/'COMPLETE.json').exists():doc.add_paragraph(f'{variant}：等待完整结果。');continue
        policy=json.loads((root/'protocol.json').read_text());lock=json.loads((root/'locked_final.json').read_text())
        oof=pd.read_csv(root/'oof_predictions.csv')
        ext=pd.read_csv(root/'final/predictions.csv')
        labels=pd.read_csv(root/'evaluation_labels.csv').query("cohort=='external_test'")
        assert len(ext)==len(labels) and set(ext.ID)==set(labels.ID)
        ext=ext.merge(labels[['ID','event']],on='ID',validate='one_to_one')
        signature=sha(root/'oof_predictions.csv')+sha(root/'final/predictions.csv')+sha(root/'evaluation_labels.csv')+sha(__file__)
        cache=root/'statistics.json'
        if cache.exists() and json.loads(cache.read_text())['signature']==signature:
            saved=json.loads(cache.read_text());rows=saved['metrics'];ps=saved['pairs']
        else:
            rows=[];ps=[]
            for cohort,data in [('OOF',oof),('external',ext)]:
                for name in OUTPUTS:
                    rows.append({'variant':variant,'cohort':cohort,'model':name,**metrics_ci(data.event,data[name],None if cohort=='OOF' else lock['thresholds'][name],rounds=policy['statistics']['patient_stratified_bootstrap'])})
                pp=[]
                for a,b in [('CMR_CatBoost','Rscore_CMR_CatBoost'),('Rscore_CMR_EN','Rscore_CMR_CatBoost'),('CMR_EN','Selected')]:
                    pp.append({'variant':variant,'cohort':cohort,'A':a,'B':b,**paired(data.event,data[a],data[b],rounds=policy['statistics']['patient_stratified_bootstrap'])})
                for row,p in zip(pp,holm([x['DeLong_P'] for x in pp])):row['Holm_P']=float(p)
                ps+=pp
            write_json(cache,{'signature':signature,'metrics':rows,'pairs':ps})
        metrics+=rows;pairs+=ps
        doc.add_heading(variant,1)
        table=doc.add_table(rows=1,cols=4);table.style='Light Shading Accent 1'
        for c,v in zip(table.rows[0].cells,['队列','模型','AUC [95% CI]','Brier']):c.text=v
        for row in rows:
            for c,v in zip(table.add_row().cells,[row['cohort'],row['model'],f"{row['AUC']:.3f} [{row['AUC_CI_low']:.3f}, {row['AUC_CI_high']:.3f}]",f"{row['Brier']:.3f}"]):c.text=v
        for cohort,data in [('OOF',oof),('external',ext)]:
            predictions.append(data.assign(variant=variant,cohort=cohort))
            fig,axes=plt.subplots(1,2,figsize=(10,4.5))
            for name in ['CMR_CatBoost','Rscore_CMR_EN','Rscore_CMR_CatBoost','Selected']:
                fpr,tpr,_=roc_curve(data.event,data[name]);axes[0].plot(fpr,tpr,label=name)
                obs,pr=calibration_curve(data.event,data[name],n_bins=8,strategy='quantile');axes[1].plot(pr,obs,'o-',ms=3,label=name)
            for ax in axes:ax.plot([0,1],[0,1],'k--',lw=.7);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend(fontsize=6)
            axes[0].set(xlabel='1 - specificity',ylabel='Sensitivity');axes[1].set(xlabel='Estimated probability',ylabel='Observed LGE fraction')
            fig.suptitle(f'{variant}: {cohort}');fig.tight_layout()
            for suffix in ['png','pdf','tiff']:fig.savefig(out/f'{variant}_{cohort}.{suffix}',dpi=300,bbox_inches='tight')
            plt.close(fig)
            doc.add_picture(str(out/f'{variant}_{cohort}.png'),width=Inches(6.2))
        for path in root.glob('fold_*/tuning.json'):
            for row in json.loads(path.read_text())['records']:tuning.append({'variant':variant,'fold':path.parent.name,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}})
        for path in root.glob('fold_*/complete.json'):
            row=json.loads(path.read_text());timing.append({'variant':variant,'fold':path.parent.name,'seconds':row['seconds'],'selected_model':row['selected_model']})
        for path in root.rglob('selection.json'):
            row=json.loads(path.read_text());selections.append({'variant':variant,'path':str(path.relative_to(root)),**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}})
        for path in root.rglob('coefficients.csv'):
            frame=pd.read_csv(path);frame['variant']=variant;frame['path']=str(path.relative_to(root));coefficients.append(frame)
    sheets={'Metrics':pd.DataFrame(metrics),'Comparisons':pd.DataFrame(pairs),'Predictions':pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(),
            'Classifier_tuning':pd.DataFrame(tuning),'Rscore_CMR_selection':pd.DataFrame(selections),'Execution':pd.DataFrame(timing),
            'Coefficients':pd.concat(coefficients,ignore_index=True) if coefficients else pd.DataFrame()}
    with pd.ExcelWriter(out/'full_validation_results.xlsx',engine='openpyxl') as writer:
        from openpyxl.styles import Font,PatternFill
        for name,frame in sheets.items():
            frame.to_excel(writer,sheet_name=name,index=False);ws=writer.sheets[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D')
            for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=22
    doc.add_paragraph('OOF报告AUC、Brier及校准；操作阈值来自开发OOF，外部固定应用。置信区间以患者为单位对现有预测bootstrap，未重新训练全部模型。无形状分支在任何监督选择前剔除全部shape/shape2D特征。实际结论按效应量和置信区间填写，不预设新实验必须优于旧结果。')
    doc.save(out/'全流程验证结果_中文版.docx')
    write_json(out/'report_status.json',{'time':now(),'completed_variants':[v for v in ['all_features','no_shape'] if (BASE/v/'COMPLETE.json').exists()]})

if __name__=='__main__':main()
