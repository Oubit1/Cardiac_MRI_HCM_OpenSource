"""Freeze the new analysis before any new heldout model results are inspected."""
from settings import *
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import StratifiedKFold

def main():
    if (ROOT/'plans/protocol.json').exists():raise RuntimeError('Protocol already exists; refusing to rewrite existing inputs or splits.')
    for name in ['inputs','plans','runs','logs','audit','tables','figures','response']:(ROOT/name).mkdir(exist_ok=True)
    canonical=pd.read_csv(CORE/'inputs/analysis_manifest.csv')[['cohort','ID','event']]
    matched=pd.DataFrame(json.loads((CORE/'inputs/prepared_patients.json').read_text()))[['cohort','ID','event']]
    canonical=matched[['cohort','ID']].merge(canonical,on=['cohort','ID'],validate='one_to_one')
    assert np.array_equal(canonical.event.to_numpy(),matched.event.to_numpy())
    candidates=['sex','age','LVEDV','EF','HR','LVmass','MWT','maxLAV','LAEF','LAs','SRs','SRe','Sra','LVGRS','LVGCS','LVGLS']
    icc=PROJECT/'Radiomics/ICC/Radiomics_ICC_selected.csv'
    features=[x for x in pd.read_csv(icc,nrows=0).columns if x not in ['ID','Patient','name','event']]
    assert len(features)>1 and not set(features)&set(candidates)
    sources=[icc,CORE/'inputs/analysis_manifest.csv',PROJECT/'26.9.19/Rscore_nested_OOF_elasticnet.R',PROJECT/'26.9.19/CMR_parameter_LASSO.R']
    audits=[];tables={}
    for cohort,suffix in [('development','train'),('external_test','external')]:
        truth=canonical[canonical.cohort==cohort].set_index('ID')
        cmr_path=PROJECT/f'CMR parameter/{suffix}.xlsx'
        rad_path=PROJECT/f'Radiomics/Radiomics_{suffix}.xlsx'
        sources += [cmr_path,rad_path]
        cmr=pd.read_excel(cmr_path).set_index('ID');rad=pd.read_excel(rad_path).set_index('ID')
        for name,data in [('CMR',cmr),('radiomics',rad)]:
            assert not data.index.duplicated().any()
            assert set(truth.index)<=set(data.index)
            different=data.loc[truth.index,'event'].to_numpy()!=truth.event.to_numpy()
            audits.append({'cohort':cohort,'source':name,'original_n':len(data),'matched_n':len(truth),
                           'excluded_ids':sorted(set(data.index)-set(truth.index)),
                           'source_label_discrepancy_ids':truth.index[different].tolist()})
        values={'event':truth.event.astype(int)}
        values.update({'cmr__'+col:pd.to_numeric(cmr.loc[truth.index,col],errors='raise') for col in candidates})
        values.update({'rad__'+col:pd.to_numeric(rad.loc[truth.index,col],errors='raise') for col in features})
        out=pd.DataFrame(values,index=truth.index)
        out=out.replace([np.inf,-np.inf],np.nan)
        assert not out.event.isna().any()
        tables[cohort]=out
    dev=tables['development'];ext=tables['external_test']
    assert len(dev)==1050 and len(ext)==309
    dev.to_csv(ROOT/'inputs/development.csv',index_label='ID')
    # Outcome-free external inference input. External labels live only in the evaluation file.
    ext.drop(columns='event').to_csv(ROOT/'inputs/external_features.csv',index_label='ID')
    canonical.to_csv(ROOT/'inputs/evaluation_labels.csv',index=False)
    folds=[];seen=[]
    for fold in range(1,6):
        split=json.loads((CORE/f'plans/fold_{fold}_ids.json').read_text())
        tr,te=split['outer_train'],split['outer_test'];assert not set(tr)&set(te)
        assert set(tr)|set(te)==set(dev.index)
        checkpoint=CORE/f'runs/oof_extended60_full_fold{fold}/final.pt'
        cfg=json.loads((checkpoint.parent/'config.json').read_text())
        assert cfg['train_ids']==tr and not cfg['validation_ids']
        assert cfg['encoder_lr']==1e-5 and cfg['mode']=='full'
        directory=ROOT/'runs'/f'fold_{fold}';directory.mkdir(exist_ok=True)
        dev.loc[tr].to_csv(directory/'training.csv',index_label='ID')
        dev.loc[te].drop(columns='event').to_csv(directory/'prediction_features.csv',index_label='ID')
        crossfit=[]
        for j,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=8100+fold).split(tr,dev.loc[tr].event),1):
            crossfit.append({'split':j,'training_ids':np.array(tr)[a].tolist(),'heldout_ids':np.array(tr)[b].tolist()})
        spec={'fold':fold,'outer_train':tr,'outer_test':te,'rscore_crossfit':crossfit,
              'image_checkpoint':str(checkpoint),'image_checkpoint_sha256':sha(checkpoint),
              'image_config_sha256':sha(checkpoint.parent/'config.json')}
        write_json(directory/'split.json',spec);folds.append(spec);seen+=te
    assert len(seen)==len(set(seen))==1050
    shutil.copy2(PROJECT/'26.9.19/Rscore_nested_OOF_elasticnet.R',ROOT/'inputs/reference_Rscore.R')
    shutil.copy2(PROJECT/'26.9.19/CMR_parameter_LASSO.R',ROOT/'inputs/reference_CMR.R')
    source_hashes=[{'path':str(p),'sha256':sha(p)} for p in sources]
    protocol={'version':'nested-validation-2026-11-08-v1','locked_at':now(),
        'cohorts':{'development':1050,'external':309},'folds':folds,
        'image_branch':'Existing extended60_full, chosen in previous core protocol before its external evaluation; no new branch selection',
        'image_features':'Training representations are apparent supervised features, not cross-fitted; outer test excluded from encoder adaptation and epoch selection',
        'cmr_candidates':candidates,'cmr_selection':'LASSO alpha=1, training-only ten-fold selection, lambda.1se; selected variables enter fixed downstream classifier',
        'rscore_features':features,'icc_policy':'Frozen historical outcome-independent ICC feature list; not reselected using outcomes; upstream reliability provenance remains historical',
        'rscore_crossfit_folds':5,'tuning_folds':10,'alpha_grid':[round(i/10,1) for i in range(1,10)],
        'lambda_grid':np.logspace(1,-5,100).tolist(),
        'lambda_rule':'Largest lambda within one standard error of peak validation AUC for each alpha; choose alpha by peak AUC, then selected AUC, then higher alpha',
        'adaptation_from_R_script':'glmnet retained; all preprocessing refit within every tuning fold. Fixed label-independent lambda grid replaces data-derived glmnet grid; alpha tie-break does not use full-training nonzero count.',
        'rscore_training_scores':'Five-fold cross-fitted within each outer training set; preprocessing and alpha/lambda tuning repeated using crossfit-training patients only',
        'preprocessing':{'missing_cutoff':.20,'zero_variance':'remove','radiomics_correlation_cutoff':.95,
                         'CMR_correlation_filter':False,'imputation':'training median','scaling':'training mean/sample SD'},
        'fusion':{'models':MODELS,'C':.1,'l1_ratio':.5,'solver':'saga','max_iter':10000,'tol':1e-5,'seed':42,'tuning':False},
        'OOF_reporting':'Discrimination, Brier and calibration only; no independent OOF threshold-metric claim after selecting threshold on these same predictions',
        'threshold':'Prespecified specificity maximum at pooled development OOF sensitivity >=0.95; locked before new external inference',
        'final_rscore_training':'Use the five outer-heldout Rscore predictions as full-development cross-fitted training scores; fit full-development Rscore for external inference',
        'external_reuse':'Previously examined external cohort, including previous fusion results; new analysis is a revision sensitivity, not untouched confirmation',
        'primary_comparisons':[['CMR','Rscore_CMR'],['CMR','mViT_CMR'],['Rscore_CMR','mViT_Rscore_CMR']],
        'statistics':{'patient_stratified_bootstrap':2000,'paired_delong':True,'holm':'three primary comparisons separately per dataset'},
        'hardware':{'single_gpu_uuid':GPU_UUID,'R_worker_limit':4,'blas_threads_per_R_worker':1},'source_hashes':source_hashes}
    target=ROOT/'plans/protocol.json'
    if target.exists():raise RuntimeError('Protocol already exists; do not silently overwrite frozen analysis.')
    write_json(target,protocol);write_json(ROOT/'audit/input_alignment.json',audits)
    write_json(ROOT/'audit/partition_validation.json',{'passed':True,'unique_heldout_development_patients':1050,
            'external_outcomes_removed_from_inference_inputs':True,'all_five_checkpoint_training_ids_match':True})
    status('inputs_prepared',radiomics_features=len(features),CMR_candidates=len(candidates))

if __name__=='__main__':main()
