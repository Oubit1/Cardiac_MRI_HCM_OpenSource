"""Full 1059-patient outer/inner validation with upstream reconstruction and model selection."""
from settings import *
import numpy as np,pandas as pd,subprocess,time,joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from concurrent.futures import ThreadPoolExecutor

BASE=ROOT/'tabular_full'
FAMILIES=['CatBoost','LightGBM','SVM','XGBoost','Bagging']
OUTPUTS=['CMR_EN','Rscore_CMR_EN','CMR_CatBoost',*[f'Rscore_CMR_{x}' for x in FAMILIES],'Selected']

def candidates(family):
    # Finite, locked search; all candidates share exactly the same inner upstream fits.
    if family=='CatBoost':return [dict(depth=d,learning_rate=lr,iterations=it,l2_leaf_reg=3) for d,lr,it in [(3,.03,300),(4,.03,600),(5,.03,600),(4,.01,900)]]
    if family=='LightGBM':return [dict(num_leaves=n,learning_rate=lr,n_estimators=it,max_depth=-1) for n,lr,it in [(7,.03,300),(15,.03,500),(7,.01,800),(15,.01,800)]]
    if family=='XGBoost':return [dict(max_depth=d,learning_rate=lr,n_estimators=it) for d,lr,it in [(2,.03,300),(3,.03,500),(2,.01,800),(4,.03,500)]]
    if family=='SVM':return [dict(C=c,gamma=g) for c,g in [(0.1,'scale'),(1,'scale'),(10,'scale'),(1,.1)]]
    if family=='Bagging':return [dict(n_estimators=n,max_samples=s,max_features=1.) for n,s in [(100,.7),(200,.7),(100,1.),(200,1.)]]
    raise ValueError(family)

def estimator(family,params,seed=42):
    if family=='CatBoost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**params,random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False,loss_function='Logloss')
    elif family=='LightGBM':
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(**params,random_state=seed,n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True)
    elif family=='XGBoost':
        from xgboost import XGBClassifier
        model=XGBClassifier(**params,random_state=seed,n_jobs=1,tree_method='hist',eval_metric='logloss')
    elif family=='SVM':
        from sklearn.svm import SVC
        model=SVC(**params,probability=True,random_state=seed)
    elif family=='Bagging':
        from sklearn.ensemble import BaggingClassifier
        model=BaggingClassifier(**params,random_state=seed,n_jobs=1)
    elif family=='EN':model=LogisticRegression(C=.1,penalty='elasticnet',l1_ratio=.5,solver='saga',max_iter=10000,tol=1e-5,random_state=seed)
    else:raise ValueError(family)
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),model)

def split_crossfit(data,k,seed):
    return [{'split':j,'training_ids':data.ID.iloc[a].tolist(),'heldout_ids':data.ID.iloc[b].tolist()}
       for j,(a,b) in enumerate(StratifiedKFold(k,shuffle=True,random_state=seed).split(data,data.event),1)]

def prepare():
    BASE.mkdir(exist_ok=True)
    manifest=pd.read_csv(CORE/'inputs/analysis_manifest.csv')
    reference=json.loads((ROOT/'plans/protocol.json').read_text())
    for variant in ['all_features','no_shape']:
        root=BASE/variant;root.mkdir(exist_ok=True)
        if (root/'protocol.json').exists():continue
        radiomics=reference['rscore_features']
        if variant=='no_shape':radiomics=[x for x in radiomics if '_shape_' not in x.lower() and '_shape2d_' not in x.lower()]
        assert len(radiomics)>0
        datasets={}
        for cohort,suffix in [('development','train'),('external_test','external')]:
            truth=manifest.query('cohort==@cohort')[['ID','event']].set_index('ID')
            cmr=pd.read_excel(PROJECT/f'CMR parameter/{suffix}.xlsx').set_index('ID')
            rad=pd.read_excel(PROJECT/f'Radiomics/Radiomics_{suffix}.xlsx').set_index('ID')
            assert not truth.index.duplicated().any() and not cmr.index.duplicated().any() and not rad.index.duplicated().any()
            assert np.array_equal(cmr.loc[truth.index,'event'],truth.event) and np.array_equal(rad.loc[truth.index,'event'],truth.event)
            data=truth.copy()
            data=pd.concat([data,cmr.loc[truth.index,reference['cmr_candidates']].add_prefix('cmr__'),rad.loc[truth.index,radiomics].add_prefix('rad__')],axis=1)
            datasets[cohort]=data.reset_index()
        assert len(datasets['development'])==1059 and len(datasets['external_test'])==309
        datasets['development'].to_csv(root/'development.csv',index=False)
        datasets['external_test'].drop(columns='event').to_csv(root/'external_features.csv',index=False)
        manifest[['cohort','ID','event']].to_csv(root/'evaluation_labels.csv',index=False)
        policy={k:reference[k] for k in ['cmr_candidates','alpha_grid','lambda_grid','tuning_folds','preprocessing','statistics']}
        policy.update(version='full-tabular-2026-11-08-v1',variant=variant,rscore_features=radiomics,
                      outer_folds=5,inner_folds=5,rscore_crossfit_folds=5,repeats=1,
                      search={f:candidates(f) for f in FAMILIES},selection='mean inner AUC, then mean specificity at sensitivity >=.95, then earlier candidate',
                      external_reuse=True,seed=42,created=now())
        write_json(root/'protocol.json',policy)
        folds=split_crossfit(datasets['development'],5,42)
        write_json(root/'folds.json',folds)
    print('Full-cohort tabular inputs prepared (1059/309), all-features and no-shape.',flush=True)

def upstream(root,directory,train,pred,seed,final=False):
    directory.mkdir(parents=True,exist_ok=True)
    signature={'protocol':sha(root/'protocol.json'),'code':sha(ROOT/'code/nested_tabular.R'),
               'train':__import__('hashlib').sha256(train.to_csv(index=False).encode()).hexdigest(),
               'pred':__import__('hashlib').sha256(pred.to_csv(index=False).encode()).hexdigest()}
    receipt=directory/'upstream_receipt.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=signature:raise RuntimeError(f'Stale upstream output: {directory}')
        assert (directory/'r_complete.json').exists();return
    assert 'event' not in pred and not train.ID.duplicated().any() and not pred.ID.duplicated().any()
    if not final:assert set(train.ID).isdisjoint(pred.ID)
    train.to_csv(directory/'training.csv',index=False);pred.to_csv(directory/'prediction_features.csv',index=False)
    k=json.loads((root/'protocol.json').read_text())['rscore_crossfit_folds']
    write_json(directory/'split.json',{'fold':seed,'outer_train':train.ID.tolist(),'outer_test':pred.ID.tolist(),
              'rscore_crossfit':split_crossfit(train,k,seed)})
    command=[str(RSCRIPT),str(ROOT/'code/nested_tabular.R'),str(directory),str(root/'protocol.json')]
    if final:command.append('final')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with (directory/'R.log').open('a') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(receipt,signature)

def matrices(directory,with_rscore=True):
    train=pd.read_csv(directory/'training.csv').set_index('ID');pred=pd.read_csv(directory/'prediction_features.csv').set_index('ID')
    selected=json.loads((directory/'cmr_selection/selection.json').read_text())['selected_features']
    if isinstance(selected,str):selected=[selected]
    a=train[selected].copy();b=pred[selected].copy()
    if with_rscore:
        r=pd.read_csv(directory/'rscore_training_crossfit.csv').set_index('ID')
        q=pd.read_csv(directory/'rscore_prediction.csv').set_index('ID')
        assert set(r.index)==set(a.index) and set(q.index)==set(b.index)
        a['Rscore']=r.loc[a.index,'Rscore'];b['Rscore']=q.loc[b.index,'Rscore']
    if a.shape[1]==0:a['constant']=0.;b['constant']=0.
    return a.to_numpy(),train.event.to_numpy(),b.to_numpy(),pred.index.to_numpy()

def tune(root,parent,train,seed):
    policy=json.loads((root/'protocol.json').read_text());folds=split_crossfit(train,policy['inner_folds'],seed)
    inner=[]
    for fold in folds:
        d=parent/f'inner_{fold["split"]}';indexed=train.set_index('ID')
        tr=indexed.loc[fold['training_ids']].reset_index();va=indexed.loc[fold['heldout_ids']].reset_index()
        upstream(root,d,tr,va.drop(columns='event'),seed+fold['split'])
        inner.append((d,va.event.to_numpy()))
    records=[];best={}
    for label,family,with_rscore in [(f'Rscore_CMR_{f}',f,True) for f in FAMILIES]+[('CMR_CatBoost','CatBoost',False)]:
        scores=[]
        for index,params in enumerate(policy['search'][family]):
            aucs=[];specs=[]
            for d,y in inner:
                x,yt,z,_=matrices(d,with_rscore);model=estimator(family,params,seed)
                model.fit(x,yt);p=model.predict_proba(z)[:,1]
                aucs.append(roc_auc_score(y,p));fpr,tpr,_=roc_curve(y,p);specs.append(float((1-fpr[tpr>=.95]).max()))
            row={'model':label,'candidate':index,'mean_auc':float(np.mean(aucs)),'mean_specificity':float(np.mean(specs)),
                 'fold_auc':aucs,'parameters':params}
            scores.append(row);records.append(row)
        best[label]=max(scores,key=lambda r:(r['mean_auc'],r['mean_specificity'],-r['candidate']))
    winner=max([f'Rscore_CMR_{f}' for f in FAMILIES],key=lambda name:(best[name]['mean_auc'],best[name]['mean_specificity'],-FAMILIES.index(name.removeprefix('Rscore_CMR_'))))
    write_json(parent/'tuning.json',{'records':records,'best':best,'selected_model':winner,'fit_ids':train.ID.tolist(),
                                  'inner_partitions':folds})
    return best,winner

def predict_models(directory,best,winner):
    output=None;models={}
    for name in OUTPUTS:
        if name=='Selected':continue
        family='EN' if name.endswith('_EN') else name.removeprefix('Rscore_CMR_').removeprefix('CMR_')
        x,y,z,ids=matrices(directory,name.startswith('Rscore'))
        model=estimator(family,{} if family=='EN' else best[name]['parameters'])
        model.fit(x,y)
        if output is None:output=pd.DataFrame({'ID':ids})
        output[name]=model.predict_proba(z)[:,1];models[name]=model
    output['Selected']=output[winner];assert np.isfinite(output[OUTPUTS]).all().all()
    output.to_csv(directory/'predictions.csv',index=False);joblib.dump(models,directory/'models.joblib')

def fingerprint(root):
    return {'protocol':sha(root/'protocol.json'),'code':sha(__file__),'R_code':sha(ROOT/'code/nested_tabular.R'),
            'development':sha(root/'development.csv'),'external_features':sha(root/'external_features.csv'),'folds':sha(root/'folds.json')}

def outer(root,fold):
    d=root/f'fold_{fold}';d.mkdir(exist_ok=True);signature=fingerprint(root)
    if (d/'complete.json').exists():
        if json.loads((d/'complete.json').read_text())['fingerprint']!=signature:raise RuntimeError(f'Stale completed fold: {d}')
        return
    began=time.monotonic();data=pd.read_csv(root/'development.csv').set_index('ID')
    split=json.loads((root/'folds.json').read_text())[fold-1]
    tr=data.loc[split['training_ids']].reset_index();te=data.loc[split['heldout_ids']].reset_index()
    best,winner=tune(root,d,tr,11000+100*fold)
    upstream(root,d/'outer',tr,te.drop(columns='event'),12000+fold)
    predict_models(d/'outer',best,winner)
    write_json(d/'complete.json',{'fingerprint':signature,'seconds':time.monotonic()-began,'selected_model':winner,
                                 'training_ids':tr.ID.tolist(),'heldout_ids':te.ID.tolist()})
    print(f'{root.name} fold {fold} complete ({time.monotonic()-began:.1f}s)',flush=True)

def finalize(root):
    if not all((root/f'fold_{i}/complete.json').exists() for i in range(1,6)):return
    for i in range(1,6):
        if json.loads((root/f'fold_{i}/complete.json').read_text())['fingerprint']!=fingerprint(root):raise RuntimeError('Outer fold code/data changed; cannot mix outputs')
    if (root/'COMPLETE.json').exists():
        if json.loads((root/'COMPLETE.json').read_text())['fingerprint']!=fingerprint(root):raise RuntimeError('Stale final results')
        return
    data=pd.read_csv(root/'development.csv');folds=json.loads((root/'folds.json').read_text())
    # Reuse valid outer upstream matrices as inner folds for full-development tuning.
    records=[];best={}
    policy=json.loads((root/'protocol.json').read_text())
    for label,family,with_rscore in [(f'Rscore_CMR_{f}',f,True) for f in FAMILIES]+[('CMR_CatBoost','CatBoost',False)]:
        rows=[]
        for index,params in enumerate(policy['search'][family]):
            aucs=[];specs=[]
            for i in range(1,6):
                x,y,z,ids=matrices(root/f'fold_{i}/outer',with_rscore)
                truth=data.set_index('ID').loc[ids,'event'].to_numpy()
                model=estimator(family,params);model.fit(x,y);p=model.predict_proba(z)[:,1]
                aucs.append(roc_auc_score(truth,p));fpr,tpr,_=roc_curve(truth,p);specs.append(float((1-fpr[tpr>=.95]).max()))
            row={'model':label,'candidate':index,'mean_auc':float(np.mean(aucs)),'mean_specificity':float(np.mean(specs)),'parameters':params}
            rows.append(row);records.append(row)
        best[label]=max(rows,key=lambda r:(r['mean_auc'],r['mean_specificity'],-r['candidate']))
    winner=max([f'Rscore_CMR_{f}' for f in FAMILIES],key=lambda n:(best[n]['mean_auc'],best[n]['mean_specificity']))
    oof=pd.concat([pd.read_csv(root/f'fold_{i}/outer/predictions.csv').assign(fold=i) for i in range(1,6)],ignore_index=True)
    assert len(oof)==len(data) and not oof.ID.duplicated().any() and set(oof.ID)==set(data.ID)
    oof=oof.merge(data[['ID','event']],on='ID',validate='one_to_one');oof.to_csv(root/'oof_predictions.csv',index=False)
    cuts={}
    for name in OUTPUTS:
        fpr,tpr,cut=roc_curve(oof.event,oof[name],drop_intermediate=False);valid=np.flatnonzero((tpr>=.95)&np.isfinite(cut))
        cutidx=valid[np.isclose(fpr[valid],fpr[valid].min())];cuts[name]=float(cut[cutidx].max())
    locked={'thresholds':cuts,'selected_model':winner,'best':best,'records':records,'fingerprint':fingerprint(root)}
    path=root/'locked_final.json'
    if path.exists() and json.loads(path.read_text())!=locked:raise RuntimeError('Final lock changed')
    write_json(path,locked) # Before external fitting/prediction; external outcomes absent.
    d=root/'final';pred=pd.read_csv(root/'external_features.csv')
    upstream(root,d,data,pred,15000,final=True)
    scores=pd.concat([pd.read_csv(root/f'fold_{i}/outer/rscore_prediction.csv')[['ID','Rscore']] for i in range(1,6)])
    scores.set_index('ID').loc[data.ID].to_csv(d/'rscore_training_crossfit.csv',index_label='ID')
    predict_models(d,best,winner)
    write_json(root/'COMPLETE.json',{'fingerprint':fingerprint(root),'time':now(),'external_reused':True})

def run(variant,workers=2,fold_numbers=(1,2,3,4,5)):
    root=BASE/variant
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(lambda f:outer(root,f),fold_numbers))
    finalize(root)
