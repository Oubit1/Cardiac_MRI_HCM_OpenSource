"""CPU integration and partition tests on synthetic data, excluded from science outputs."""
from settings import *
import subprocess,shutil,time
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold

def main():
    protocol=json.loads((ROOT/'plans/protocol.json').read_text())
    directory=ROOT/'audit/software_test';directory.mkdir(exist_ok=True)
    rng=np.random.default_rng(1701);n=150;m=24
    x=rng.normal(size=(n+m,12));y=(x[:n,0]+rng.normal(size=n)>.0).astype(int)
    train=pd.DataFrame(x[:n],columns=[f'rad__f{i}' for i in range(12)])
    pred=pd.DataFrame(x[n:],columns=train.columns)
    for j in range(4):train[f'cmr__c{j}']=x[:n,j];pred[f'cmr__c{j}']=x[n:,j]
    train['ID']=np.arange(1,n+1);pred['ID']=np.arange(n+1,n+m+1);train['event']=y
    # Exercise missingness/variance/correlation handling using training data only.
    train.loc[:5,'rad__f4']=np.nan;train['rad__f5']=train['rad__f0'];train['rad__f6']=1.
    pred['rad__f5']=pred['rad__f0'];pred['rad__f6']=1.
    train.to_csv(directory/'training.csv',index=False);pred.to_csv(directory/'prediction_features.csv',index=False)
    cross=[]
    for i,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=7).split(x[:n],y),1):
        cross.append({'split':i,'training_ids':train.ID.iloc[a].tolist(),'heldout_ids':train.ID.iloc[b].tolist()})
    write_json(directory/'split.json',{'fold':1,'outer_train':train.ID.tolist(),'outer_test':pred.ID.tolist(),'rscore_crossfit':cross})
    protocol.update(rscore_features=[f'f{i}' for i in range(12)],cmr_candidates=[f'c{i}' for i in range(4)],
                    tuning_folds=3,alpha_grid=[.1,.5,.9],lambda_grid=np.logspace(1,-4,30).tolist())
    write_json(directory/'test_protocol.json',protocol)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',TZ='Asia/Shanghai')
    began=time.monotonic()
    subprocess.run([str(RSCRIPT),str(ROOT/'code/nested_tabular.R'),str(directory),str(directory/'test_protocol.json')],env=env,check=True)
    for child in [directory/'rscore_full',directory/'cmr_selection',*[directory/f'rscore_crossfit_{i}' for i in range(1,6)]]:
        selected=json.loads((child/'selection.json').read_text())
        ids=set(selected['training_ids']);assert ids<=set(train.ID) and not ids&set(pred.ID)
        for boundaries in json.loads((child/'preprocessing_boundaries.json').read_text()):
            assert set(boundaries['fit_ids']).isdisjoint(boundaries['validation_ids'])
            assert set(boundaries['fit_ids'])|set(boundaries['validation_ids'])==ids
        if child.name.startswith('rscore_crossfit_'):
            j=int(child.name.rsplit('_',1)[1]);assert ids==set(cross[j-1]['training_ids'])
            assert not ids&set(cross[j-1]['heldout_ids'])
    assert len(pd.read_csv(directory/'rscore_training_crossfit.csv'))==n
    vectors=rng.normal(size=(n+m,256));vectors[:,0]=x[:,0]
    for role,ids,z in [('train',train.ID,vectors[:n]),('heldout',pred.ID,vectors[n:])]:
        np.savez_compressed(directory/f'image_{role}.npz',ID=ids,features=z)
    for role,ids in [('train',train.ID),('heldout',pred.ID)]:
        np.savez_compressed(directory/f'frozen_{role}.npz',ID=ids,features=rng.normal(size=(len(ids),512)))
    from fusion_stage import run_fusion
    run_fusion(directory)
    p=pd.read_csv(directory/'predictions.csv');assert p.ID.tolist()==pred.ID.tolist()
    assert p[MODELS].ge(0).all().all() and p[MODELS].le(1).all().all()
    bad=directory/'rejected_label_input';bad.mkdir(exist_ok=True)
    shutil.copy2(directory/'training.csv',bad/'training.csv')
    pred.assign(event=0).to_csv(bad/'prediction_features.csv',index=False)
    rejected=subprocess.run([str(RSCRIPT),str(ROOT/'code/nested_tabular.R'),str(bad),str(directory/'test_protocol.json')],
                             env=env,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'event' in rejected.stderr
    write_json(ROOT/'audit/software_validation.json',{'passed':True,'time':now(),'seconds':time.monotonic()-began,
       'role':'Synthetic software-only test; not scientific results','checks':[
       'R glmnet tuning and lambda.1se selection','preprocessing fitted inside each tuning fold',
       'Rscore crossfit excludes target patient outcomes throughout tuning','CMR selection excludes outer heldout',
       'heldout event column explicitly rejected','all nine fixed fusion classifiers fit and predict',
       'ID order and full patient coverage','finite probabilities'],
       'r_code_sha256':sha(ROOT/'code/nested_tabular.R'),'fusion_code_sha256':sha(ROOT/'code/fusion_stage.py')})
    print('CPU software and partition validation passed.',flush=True)

if __name__=='__main__':main()
