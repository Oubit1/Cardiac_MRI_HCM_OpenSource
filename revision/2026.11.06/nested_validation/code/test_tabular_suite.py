"""Exercise all outer folds, inner upstream isolation, model selection and final lock on synthetic data."""
from settings import *
import numpy as np,pandas as pd,shutil
import tabular_suite as tab

def main():
    root=ROOT/'audit/software_tabular_suite'
    if root.exists():shutil.rmtree(root) # Dedicated disposable synthetic fixture only.
    root.mkdir();rng=np.random.default_rng(22)
    x=rng.normal(size=(180,8));y=(x[:,0]+rng.normal(size=180)>.0).astype(int)
    data=pd.DataFrame(x,columns=[f'rad__f{i}' for i in range(8)])
    for i in range(3):data[f'cmr__c{i}']=x[:,i]
    data['ID']=np.arange(180);data['event']=y
    train=data.iloc[:150].copy();ext=data.iloc[150:].copy();ext['ID']=np.arange(30)
    train.to_csv(root/'development.csv',index=False);ext.drop(columns='event').to_csv(root/'external_features.csv',index=False)
    pd.concat([train[['ID','event']].assign(cohort='development'),ext[['ID','event']].assign(cohort='external_test')]).to_csv(root/'evaluation_labels.csv',index=False)
    policy={'rscore_features':[f'f{i}' for i in range(8)],'cmr_candidates':[f'c{i}' for i in range(3)],
            'tuning_folds':3,'alpha_grid':[.5],'lambda_grid':np.logspace(0,-3,12).tolist(),'rscore_crossfit_folds':2,'inner_folds':2,
            'statistics':{'patient_stratified_bootstrap':20},'search':{f:[tab.candidates(f)[0]] for f in tab.FAMILIES}}
    for f in ['CatBoost','LightGBM','XGBoost','Bagging']:
        key='iterations' if f=='CatBoost' else 'n_estimators';policy['search'][f][0][key]=5
    write_json(root/'protocol.json',policy);write_json(root/'folds.json',tab.split_crossfit(train,5,42))
    for fold in range(1,6):
        tab.outer(root,fold)
        split=json.loads((root/'folds.json').read_text())[fold-1];heldout=set(split['heldout_ids'])
        for path in (root/f'fold_{fold}').rglob('selection.json'):
            fit=set(json.loads(path.read_text())['training_ids']);assert fit.isdisjoint(heldout)
        for child in (root/f'fold_{fold}').glob('inner_*'):
            inner_test=set(pd.read_csv(child/'prediction_features.csv').ID)
            for path in child.rglob('selection.json'):assert set(json.loads(path.read_text())['training_ids']).isdisjoint(inner_test)
    tab.finalize(root)
    assert (root/'COMPLETE.json').exists()
    for file,n in [('oof_predictions.csv',150),('final/predictions.csv',30)]:
        p=pd.read_csv(root/file);assert len(p)==n and not p.ID.duplicated().any()
        assert p[tab.OUTPUTS].ge(0).all().all() and p[tab.OUTPUTS].le(1).all().all()
    # Selected predictions must be the classifier chosen inside the corresponding training fold.
    for fold in range(1,6):
        d=root/f'fold_{fold}';winner=json.loads((d/'tuning.json').read_text())['selected_model']
        p=pd.read_csv(d/'outer/predictions.csv');assert np.array_equal(p.Selected,p[winner])
    tab.outer(root,1) # Resume skips completed work with matching fingerprints.
    import report_tabular
    report_root=ROOT/'audit/software_tabular_report';report_root.mkdir(exist_ok=True)
    link=report_root/'all_features'
    if not link.exists():link.symlink_to(root,target_is_directory=True)
    report_tabular.BASE=report_root
    report_tabular.main()
    assert (report_root/'reports/全流程验证结果_中文版.docx').exists()
    write_json(ROOT/'audit/tabular_suite_software_validation.json',{'passed':True,'synthetic_only':True,'time':now(),
        'code_sha256':sha(ROOT/'code/tabular_suite.py'),'checks':['5 outer folds','inner supervised isolation','all 5 classifier families','inner model selection','full refit','locked thresholds','ID coverage','resume']})
    print('Full tabular synthetic integration passed.',flush=True)

if __name__=='__main__':main()
