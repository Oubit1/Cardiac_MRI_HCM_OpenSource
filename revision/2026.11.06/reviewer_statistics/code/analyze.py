"""Post-training reviewer analyses; patient-level matching and fixed development thresholds."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import fisher_exact,mannwhitneyu,spearmanr,binomtest
from sklearn.metrics import roc_curve
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent;NEST=CORE/'nested_validation'
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import metrics_ci,paired,holm
ROUNDS=2000
TABLES={}

def save(name,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(ROOT/f'tables/{name}.csv',index=False);TABLES[name]=frame;return frame

def threshold(y,p):
    fpr,tpr,t=roc_curve(y,p,drop_intermediate=False);v=np.flatnonzero((tpr>=.95)&np.isfinite(t));b=v[np.isclose(fpr[v],min(fpr[v]))]
    return float(max(t[b]))

def bootstrap(y,fn):
    y=np.asarray(y);rng=np.random.default_rng(42);ixs=[np.flatnonzero(y==v) for v in [0,1]]
    out=np.array([fn(np.concatenate([rng.choice(ix,len(ix)) for ix in ixs])) for _ in range(ROUNDS)])
    return np.nanquantile(out,[.025,.975],axis=0)

def prepare():
    datasets={};cuts={};sources=[]
    for variant,prefix in [('all_features','full'),('no_shape','texture')]:
        r=NEST/'tabular_full'/variant;labels=pd.read_csv(r/'evaluation_labels.csv')
        for cohort in ['development','external_test']:
            if cohort=='development':
                probs=pd.read_csv(r/'oof_predictions.csv')
                scores=pd.concat([pd.read_csv(r/f'fold_{f}/outer/rscore_prediction.csv').assign(score_fold=f) for f in range(1,6)],ignore_index=True)
                sources += [r/f'fold_{f}/outer/rscore_prediction.csv' for f in range(1,6)]
            else:
                probs=pd.read_csv(r/'final/predictions.csv');scores=pd.read_csv(r/'final/rscore_prediction.csv')
                sources += [r/'final/predictions.csv',r/'final/rscore_prediction.csv']
            probs=probs.drop(columns=['event'],errors='ignore').merge(labels.query('cohort==@cohort')[['ID','event']],on='ID',validate='one_to_one')
            assert set(probs.ID)==set(scores.ID) and not probs.ID.duplicated().any()
            data=probs.merge(scores,on='ID',validate='one_to_one')
            if prefix=='full':
                data=data.rename(columns={'Rscore':'full_score','Rscore_probability':'Rscore_full','Rscore_CMR_CatBoost':'CMR_Rscore_full','CMR_CatBoost':'CMR','Rscore_CMR_EN':'CMR_Rscore_EN'})
                datasets[cohort]=data
            else:
                keep=data[['ID','event','Rscore','Rscore_probability','Rscore_CMR_CatBoost']].rename(columns={'Rscore':'texture_score','Rscore_probability':'Rscore_texture','Rscore_CMR_CatBoost':'CMR_Rscore_texture'})
                joined=datasets[cohort].merge(keep,on=['ID','event'],validate='one_to_one');assert len(joined)==len(data);datasets[cohort]=joined
        lock=json.loads((r/'locked_final.json').read_text())
        cuts['CMR_Rscore_full' if prefix=='full' else 'CMR_Rscore_texture']=lock['thresholds']['Rscore_CMR_CatBoost']
        if prefix=='full':
            cuts['CMR']=lock['thresholds']['CMR_CatBoost'];cuts['CMR_Rscore_EN']=lock['thresholds']['Rscore_CMR_EN'];cuts['Selected']=lock['thresholds']['Selected']
    for name in ['Rscore_full','Rscore_texture']:cuts[name]=threshold(datasets['development'].event,datasets['development'][name])
    for cohort,data in datasets.items():save('predictions_'+cohort,data)
    (ROOT/'inputs/thresholds.json').write_text(json.dumps(cuts,indent=2))
    sources += [CORE/'inputs/acquisition_metadata.csv',NEST/'tables/nested_validation_results.xlsx']
    (ROOT/'audit/source_hashes.json').write_text(json.dumps([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],indent=2))
    return datasets,cuts

def discrimination(datasets,cuts):
    names=['CMR','Rscore_full','Rscore_texture','CMR_Rscore_full','CMR_Rscore_texture','CMR_Rscore_EN','Selected']
    rows=[];comparisons=[]
    for cohort,data in datasets.items():
        for name in names:rows.append({'cohort':cohort,'model':name,**metrics_ci(data.event,data[name],None if cohort=='development' else cuts[name],rounds=ROUNDS)})
        ps=[]
        for a,b in [('Rscore_texture','Rscore_full'),('CMR_Rscore_texture','CMR_Rscore_full'),('CMR','CMR_Rscore_texture'),('CMR','CMR_Rscore_full')]:
            ps.append({'cohort':cohort,'A':a,'B':b,**paired(data.event,data[a],data[b],rounds=ROUNDS)})
        for row,p in zip(ps,holm([r['DeLong_P'] for r in ps])):row['Holm_P']=p
        comparisons+=ps
    save('performance',rows);save('shape_comparisons',comparisons)

def reclassification(datasets,cuts):
    records=[];operating=[];transitions=[]
    for cohort,data in datasets.items():
        y=data.event.to_numpy();a=data.CMR.to_numpy()
        for name in ['CMR_Rscore_full','CMR_Rscore_texture']:
            b=data[name].to_numpy();change=np.sign(b-a)
            def calc(ix):
                yy=y[ix];s=change[ix];diff=(b-a)[ix]
                e=s[yy==1].mean();n=-s[yy==0].mean()
                return [e,n,e+n,diff[yy==1].mean()-diff[yy==0].mean()]
            values=calc(np.arange(len(y)));ci=bootstrap(y,calc)
            for j,metric in enumerate(['NRI_event','NRI_nonevent','NRI_total','IDI']):records.append({'cohort':cohort,'A':'CMR','B':name,'metric':metric,'estimate':values[j],'CI_low':ci[0,j],'CI_high':ci[1,j]})
            for event in [0,1]:
                for direction in [-1,0,1]:transitions.append({'cohort':cohort,'B':name,'event':event,'probability_direction':direction,'n':int(((y==event)&(change==direction)).sum())})
            if cohort!='external_test':continue
            pa=a>=cuts['CMR'];pb=b>=cuts[name]
            def delta(ix):
                yy=y[ix];aa=pa[ix];bb=pb[ix]
                return [(~bb[yy==0]).mean()-(~aa[yy==0]).mean(),bb[yy==1].mean()-aa[yy==1].mean()]
            ci=bootstrap(y,delta);vals=delta(np.arange(len(y)))
            for j,event in enumerate([0,1]):
                # Paired changes in correctness within each true-outcome group.
                base=pa[y==event]==event;new=pb[y==event]==event
                improved=int((~base&new).sum());worse=int((base&~new).sum())
                operating.append({'B':name,'metric':['Specificity','Sensitivity'][j],'delta':vals[j],'CI_low':ci[0,j],'CI_high':ci[1,j],
                  'improved_n':improved,'worsened_n':worse,'McNemar_exact_P':binomtest(improved,improved+worse,.5).pvalue if improved+worse else 1.,'threshold_A':cuts['CMR'],'threshold_B':cuts[name]})
    save('NRI_IDI',records);save('reclassification_counts',transitions);save('operating_changes',operating)

def prevalence(data,cuts):
    y=data.event.to_numpy();rows=[]
    for name in ['CMR','CMR_Rscore_full','CMR_Rscore_texture']:
        positive=data[name].to_numpy()>=cuts[name];pos=positive[y==1];neg=positive[y==0]
        rng=np.random.default_rng(42)
        se_boot=np.array([rng.choice(pos,len(pos)).mean() for _ in range(ROUNDS)])
        sp_boot=np.array([1-rng.choice(neg,len(neg)).mean() for _ in range(ROUNDS)])
        se=pos.mean();sp=1-neg.mean()
        for prev in np.arange(.1,.91,.05):
            npv=sp*(1-prev)/(sp*(1-prev)+(1-se)*prev);ppv=se*prev/(se*prev+(1-sp)*(1-prev))
            bn=sp_boot*(1-prev)/(sp_boot*(1-prev)+(1-se_boot)*prev);bp=se_boot*prev/(se_boot*prev+(1-sp_boot)*(1-prev))
            nci=np.quantile(bn,[.025,.975]);pci=np.quantile(bp,[.025,.975])
            rows.append({'model':name,'prevalence':round(float(prev),2),'NPV':npv,'NPV_CI_low':nci[0],'NPV_CI_high':nci[1],'PPV':ppv,'PPV_CI_low':pci[0],'PPV_CI_high':pci[1]})
    save('prevalence_scenarios',rows)

def acquisition(datasets,cuts):
    meta=pd.read_csv(CORE/'inputs/acquisition_metadata.csv')
    meta['pixel_mm']=np.sqrt(meta.pixel_spacing_row_mm*meta.pixel_spacing_col_mm)
    mapping={'SIEMENS':'Siemens','GE MEDICAL SYSTEMS':'GE','PHILIPS':'Philips','UIH':'UIH'}
    meta['vendor']=meta.manufacturer.fillna('').str.upper().str.strip().map(mapping).fillna('Unverified/anonymized')
    cutoff=float(meta.query("cohort=='development'").pixel_mm.median())
    meta['resolution_group']=np.where(meta.pixel_mm<=cutoff,f'<= {cutoff:.4f} mm',f'> {cutoff:.4f} mm')
    audit={'pixel_definition':'sqrt(row spacing * column spacing), mm','resolution_cutoff_from_development_median':cutoff,
           'vendor_warning':'Extracted headers contain anonymized Ano and multiple external vendors, inconsistent with manuscript external Siemens-only description; vendor results exploratory pending source verification.',
           'center_mapping':'External institution missing; no center inferred from field strength or vendor.'}
    (ROOT/'audit/acquisition_policy.json').write_text(json.dumps(audit,indent=2))
    rows=[];dist=[];associations=[];regressions=[];joined=[]
    dev=datasets['development'];mean=dev.full_score.mean();sd=dev.full_score.std()
    for cohort,data in datasets.items():
        z=data.merge(meta.drop(columns='event'),on='ID',validate='one_to_one') if False else data.merge(meta.query('cohort==@cohort').drop(columns=['event']),on='ID',validate='one_to_one')
        z['Rscore_z']=(z.full_score-mean)/sd;joined.append(z)
        for factor in ['field_strength_t','vendor','resolution_group']:
            for group,sub in z.groupby(factor,dropna=False):
                dist.append({'cohort':cohort,'factor':factor,'group':group,'N':len(sub),'Events':int(sub.event.sum()),'Rscore_median':sub.Rscore_z.median(),'Rscore_Q1':sub.Rscore_z.quantile(.25),'Rscore_Q3':sub.Rscore_z.quantile(.75)})
                if cohort!='external_test':continue
                for name in ['CMR','Rscore_full','CMR_Rscore_full','Selected']:
                    if sub.event.nunique()<2:
                        rows.append({'factor':factor,'group':group,'model':name,'N':len(sub),'Events':int(sub.event.sum()),'status':'single outcome class; AUC not estimable'});continue
                    rows.append({'factor':factor,'group':group,'model':name,'status':'exploratory header-based subgroup',**metrics_ci(sub.event,sub[name],cuts[name],rounds=ROUNDS)})
        table=pd.crosstab(z.field_strength_t,z.event)
        associations.append({'cohort':cohort,'analysis':'field strength vs LGE','P':fisher_exact(table.to_numpy())[1] if table.shape==(2,2) else np.nan,'N':len(z),'note':'not estimable if only one field strength'})
        a=z.loc[z.event==0,'pixel_mm'];b=z.loc[z.event==1,'pixel_mm']
        associations.append({'cohort':cohort,'analysis':'pixel spacing vs LGE (Mann Whitney)','P':mannwhitneyu(a,b).pvalue,'N':len(z)})
        corr=spearmanr(z.pixel_mm,z.Rscore_z)
        associations.append({'cohort':cohort,'analysis':'pixel spacing vs Rscore (Spearman)','estimate':corr.statistic,'P':corr.pvalue,'N':len(z)})
        # Prespecified exploratory adjusted association; center unavailable.
        for model in ['field_pixel','verified_vendor']:
            sub=z if model=='field_pixel' else z[z.vendor!='Unverified/anonymized']
            if len(sub)<20:continue
            x=sub[['event','pixel_mm']].copy()
            if sub.field_strength_t.nunique()>1:x['field_3T']=(sub.field_strength_t==3).astype(int)
            if cohort=='development':x=pd.concat([x,pd.get_dummies(sub.score_fold,prefix='fold',drop_first=True,dtype=float)],axis=1)
            if model=='verified_vendor':
                if sub.vendor.nunique()<2:continue
                x=pd.concat([x,pd.get_dummies(sub.vendor,prefix='vendor',drop_first=True,dtype=float)],axis=1)
            x=x.astype(float);x.insert(0,'const',1.)
            if np.linalg.matrix_rank(x)<x.shape[1]:
                regressions.append({'cohort':cohort,'analysis':model,'status':'rank deficient; not estimated'});continue
            matrix=x.to_numpy();response=sub.Rscore_z.to_numpy()
            inverse=np.linalg.inv(matrix.T@matrix);beta=inverse@matrix.T@response
            residual=response-matrix@beta;leverage=np.sum((matrix@inverse)*matrix,axis=1)
            weight=(residual/np.maximum(1-leverage,1e-10))**2
            covariance=inverse@(matrix.T@(weight[:,None]*matrix))@inverse
            se=np.sqrt(np.maximum(np.diag(covariance),0));pval=2*norm.sf(np.abs(beta)/np.maximum(se,1e-15))
            for j,term in enumerate(x.columns):regressions.append({'cohort':cohort,'analysis':model,'term':term,'N':len(sub),'coefficient':beta[j],'CI_low':beta[j]-1.95996398454*se[j],'CI_high':beta[j]+1.95996398454*se[j],'P':pval[j],'status':'OLS HC3 normal CI; center unadjusted; association not causal'})
    save('subgroup_performance',rows);save('Rscore_acquisition_distribution',dist);save('acquisition_association',associations);save('Rscore_adjusted_association',regressions);save('metadata_joined',pd.concat(joined,ignore_index=True))

def clinical_templates(data,cuts):
    z=data[['ID','event','CMR_Rscore_full','Selected']].copy()
    z['FN_CatBoost']=(z.event==1)&(z.CMR_Rscore_full<cuts['CMR_Rscore_full'])
    z['FN_Selected']=(z.event==1)&(z.Selected<cuts['Selected'])
    z['CatBoost_threshold']=cuts['CMR_Rscore_full'];z['Selected_threshold']=cuts['Selected']
    for col in ['center_verified','LGE_percent_LV_mass','LGE_location','LGE_pattern','isolated_RV_insertion','extensive_LGE_ge15','clinical_risk_review','source_record','reviewer']:z[col]=''
    save('clinical_review_all_external',z);save('false_negative_review',z[z.FN_CatBoost|z.FN_Selected])

def main():
    datasets,cuts=prepare()
    print('Inputs aligned; thresholds saved.',flush=True)
    discrimination(datasets,cuts);print('Discrimination complete.',flush=True)
    reclassification(datasets,cuts);prevalence(datasets['external_test'],cuts);print('NRI and prevalence complete.',flush=True)
    acquisition(datasets,cuts);clinical_templates(datasets['external_test'],cuts)
    with pd.ExcelWriter(ROOT/'tables/统计明细.xlsx',engine='openpyxl') as w:
        for name,df in TABLES.items():df.to_excel(w,sheet_name=name[:31],index=False)
    (ROOT/'audit/statistics_complete.json').write_text(json.dumps({'complete':True,'bootstrap':ROUNDS,'patients':{k:len(v) for k,v in datasets.items()},'tables':list(TABLES)},indent=2))
    print('All post-training statistics complete.',flush=True)

if __name__=='__main__':main()
