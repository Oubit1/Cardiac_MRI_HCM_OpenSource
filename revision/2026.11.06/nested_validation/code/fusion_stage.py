"""Fixed downstream estimators: fitting is confined to the designated training set."""
from settings import *
import numpy as np,pandas as pd,joblib,warnings,time
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

def load_features(path,ids):
    with np.load(path) as f:
        frame=pd.DataFrame(f['features'],index=f['ID'])
    assert not frame.index.duplicated().any() and set(ids)==set(frame.index)
    return frame.loc[ids].to_numpy()

def run_fusion(directory,final=False):
    directory=Path(directory);began=time.monotonic()
    train=pd.read_csv(directory/'training.csv').set_index('ID')
    test=pd.read_csv(directory/'prediction_features.csv').set_index('ID')
    assert 'event' not in test
    if not final:assert not set(train.index)&set(test.index)
    spec=json.loads((directory/'split.json').read_text())
    if not final:
        assert train.index.tolist()==spec['outer_train'] and test.index.tolist()==spec['outer_test']
    selection=json.loads((directory/'cmr_selection/selection.json').read_text())
    selected=selection['selected_features']
    if isinstance(selected,str):selected=[selected]
    assert all(x.startswith('cmr__') for x in selected)
    rtrain=pd.read_csv(directory/'rscore_training_crossfit.csv').set_index('ID')
    rtest=pd.read_csv(directory/'rscore_prediction.csv').set_index('ID')
    assert set(rtrain.index)==set(train.index) and set(rtest.index)==set(test.index)
    rtr=rtrain.loc[train.index,['Rscore']].to_numpy();rte=rtest.loc[test.index,['Rscore']].to_numpy()
    itrain=load_features(directory/'image_train.npz',train.index)
    itest=load_features(directory/'image_heldout.npz',test.index)
    ftrain=load_features(directory/'frozen_train.npz',train.index)
    ftest=load_features(directory/'frozen_heldout.npz',test.index)
    output=pd.DataFrame({'ID':test.index});models={};columns={} 
    policy=json.loads((ROOT/'plans/protocol.json').read_text())['fusion']
    for model_name in MODELS:
        parts_train=[];parts_test=[];names=[]
        if 'CMR' in model_name and selected:
            parts_train.append(train[selected].to_numpy());parts_test.append(test[selected].to_numpy());names+=selected
        if 'Rscore' in model_name:
            parts_train.append(rtr);parts_test.append(rte);names+=['Rscore_crossfit_training_fullfit_prediction']
        if 'Frozen' in model_name:
            parts_train.append(ftrain);parts_test.append(ftest);names += [f'Frozen_{i:03d}' for i in range(512)]
        if 'mViT' in model_name:
            parts_train.append(itrain);parts_test.append(itest);names += [f'mViT_{i:03d}' for i in range(256)]
        if not parts_train:
            parts_train=[np.zeros((len(train),1))];parts_test=[np.zeros((len(test),1))];names=['intercept_only_dummy']
        x=np.concatenate(parts_train,axis=1);z=np.concatenate(parts_test,axis=1)
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
             LogisticRegression(C=policy['C'],penalty='elasticnet',l1_ratio=policy['l1_ratio'],
                solver=policy['solver'],max_iter=policy['max_iter'],tol=policy['tol'],random_state=policy['seed']))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            model.fit(x,train.event)
        output[model_name]=model.predict_proba(z)[:,1]
        models[model_name]=model;columns[model_name]=names
    assert np.isfinite(output[MODELS].to_numpy()).all()
    output.to_csv(directory/'predictions.csv',index=False)
    joblib.dump(models,directory/'fusion_models.joblib')
    write_json(directory/'fusion_complete.json',{'completed':True,'seconds':time.monotonic()-began,
               'training_ids':train.index.tolist(),'prediction_ids':test.index.tolist(),
               'cmr_features':selected,'feature_columns':columns,'fusion_tuning':False,
               'Rscore_training':'crossfitted within training partition',
               'mViT_training':'apparent supervised representations; not called OOF',
               'evaluation_role':'external revision sensitivity' if final else 'whole-pipeline outer-heldout evaluation'})

def prepare_final():
    directory=ROOT/'runs/final';directory.mkdir(exist_ok=True)
    dev=pd.read_csv(ROOT/'inputs/development.csv').set_index('ID')
    ext=pd.read_csv(ROOT/'inputs/external_features.csv').set_index('ID')
    dev.to_csv(directory/'training.csv',index_label='ID');ext.to_csv(directory/'prediction_features.csv',index_label='ID')
    rscore=pd.concat([pd.read_csv(ROOT/f'runs/fold_{i}/rscore_prediction.csv')[['ID','Rscore']] for i in range(1,6)])
    assert not rscore.ID.duplicated().any() and set(rscore.ID)==set(dev.index)
    rscore.set_index('ID').loc[dev.index].to_csv(directory/'rscore_training_crossfit.csv',index_label='ID')
    source=CORE/'runs/final_extended60_full'
    for name,original,ids in [('image_train.npz','development_apparent_features.npz',dev.index),
                              ('image_heldout.npz','external_predictions.npz',ext.index)]:
        path=source/original
        load_features(path,ids)
        destination=directory/name
        if not destination.exists():destination.symlink_to(path)
    write_json(directory/'split.json',{'fold':'final','training_cohort':'development','prediction_cohort':'external_test',
               'training_ids':dev.index.tolist(),'prediction_ids':ext.index.tolist(),
               'image_source':str(source),'image_checkpoint_sha256':sha(source/'final.pt')})
    from prepare_frozen import attach
    attach(directory,dev.index,ext.index,external=True)
    return directory
