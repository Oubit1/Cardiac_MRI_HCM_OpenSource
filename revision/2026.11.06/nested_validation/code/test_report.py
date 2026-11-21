"""Synthetic report integration test in audit/software_report_test only."""
from settings import *
import numpy as np,pandas as pd
import report_nested as report

def main():
    sandbox=ROOT/'audit/software_report_test';sandbox.mkdir(exist_ok=True)
    for directory in ['inputs','plans','audit','runs']:(sandbox/directory).mkdir(exist_ok=True)
    rng=np.random.default_rng(18);ids=np.arange(1050);y=(ids%3!=0).astype(int)
    labels=pd.DataFrame({'ID':ids,'event':y,'cohort':'development'})
    ext=pd.DataFrame({'ID':np.arange(309),'event':(np.arange(309)%3!=0).astype(int),'cohort':'external_test'})
    pd.concat([labels,ext]).to_csv(sandbox/'inputs/evaluation_labels.csv',index=False)
    policy={'statistics':{'patient_stratified_bootstrap':20},'primary_comparisons':[['CMR','Rscore_CMR'],['CMR','mViT_CMR'],['Rscore_CMR','mViT_Rscore_CMR']]}
    write_json(sandbox/'plans/protocol.json',policy)
    for fold in range(1,6):
        d=sandbox/f'runs/fold_{fold}';d.mkdir(exist_ok=True);ix=ids[(fold-1)*210:fold*210]
        frame=pd.DataFrame({'ID':ix})
        for name in MODELS:frame[name]=1/(1+np.exp(-(rng.normal(size=len(ix))+.7*y[ix])))
        frame.to_csv(d/'predictions.csv',index=False)
        write_json(d/'fusion_complete.json',{'synthetic':True});write_json(d/'split.json',{'outer_test':ix.tolist()})
    report.ROOT=sandbox
    # A stale old test lock is intentionally not reused across newly generated fixtures.
    for path in [sandbox/'plans/locked_thresholds.json',sandbox/'runs/final/predictions.csv',sandbox/'runs/final/fusion_complete.json']:
        if path.exists():path.unlink()
    report.lock_thresholds()
    d=sandbox/'runs/final';d.mkdir(exist_ok=True);frame=ext[['ID']].copy()
    for name in MODELS:frame[name]=1/(1+np.exp(-(rng.normal(size=len(ext))+.7*ext.event)))
    frame.to_csv(d/'predictions.csv',index=False);write_json(d/'fusion_complete.json',{'synthetic':True})
    report.main()
    assert json.loads((sandbox/'audit/report_validation.json').read_text())['completed']
    result=pd.read_excel(sandbox/'tables/nested_validation_results.xlsx',sheet_name='Metrics')
    assert len(result)==18 and result.query("cohort=='development_OOF'").Threshold.isna().all()
    # Changing OOF predictions after the lock must fail.
    path=sandbox/'runs/fold_1/predictions.csv';frame=pd.read_csv(path);frame.loc[0,'CMR']+=.001;frame.to_csv(path,index=False)
    try:report.lock_thresholds()
    except RuntimeError:pass
    else:raise AssertionError('Changed OOF predictions were accepted')
    write_json(ROOT/'audit/report_software_validation.json',{'passed':True,'synthetic_only':True,'report_code_sha256':sha(ROOT/'code/report_nested.py')})
    print('Synthetic report, coverage checks and immutable threshold lock passed.')

if __name__=='__main__':main()
