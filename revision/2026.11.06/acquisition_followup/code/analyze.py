from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
from scipy.stats import norm,chi2,spearmanr
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import holm
TECH=json.loads((ROOT/'inputs/protocol.json').read_text())['technical_factors']

def ols(y,x):
 a=x.to_numpy(float);y=np.asarray(y,float);n,k=a.shape
 if np.linalg.matrix_rank(a)!=k or n<=k+5:raise ValueError('Design rank or sample size insufficient')
 inv=np.linalg.inv(a.T@a);beta=inv@a.T@y;res=y-a@beta;h=np.sum((a@inv)*a,axis=1)
 if np.max(h)>=1-1e-8:raise ValueError('Unit leverage')
 cov=inv@(a.T@((res/(1-h))[:,None]**2*a))@inv;se=np.sqrt(np.maximum(np.diag(cov),0));pv=2*norm.sf(abs(beta)/np.maximum(se,1e-15))
 rows=[dict(term=t,coefficient=beta[i],SE_HC3=se[i],CI_low=beta[i]-1.95996398454*se[i],CI_high=beta[i]+1.95996398454*se[i],P=pv[i]) for i,t in enumerate(x.columns)]
 r2=1-np.sum(res**2)/np.sum((y-y.mean())**2)
 diag={'N':n,'parameters':k,'R2':r2,'adjusted_R2':1-(1-r2)*(n-1)/(n-k),'condition_number':float(np.linalg.cond(a)),'max_leverage':h.max()}
 return rows,diag,beta,cov

def base(g,cohort):
 x=pd.DataFrame({'intercept':np.ones(len(g)),'event':g.event.to_numpy()},index=g.index)
 if cohort=='development':x=pd.concat([x,pd.get_dummies(g.fold.astype(int),prefix='outer_fold',drop_first=True,dtype=float)],axis=1)
 return x.astype(float)

def add_factor(x,g,f):
 if f=='field_strength_t':
  assert set(g[f].unique())<={1.5,3.0};x[f]=(g[f]==3).astype(float);return 1.,'3 T versus 1.5 T'
 sd=g[f].std(ddof=1);x[f]=(g[f]-g[f].mean())/sd;return sd,'per cohort-analysis SD'

def main():
 meta=pd.read_csv(ROOT/'inputs/extended_acquisition_metadata.csv');allrows=[];missing=[];diagnostics=[];drops=[];vifs=[];correlations=[];matched=[];scales=[]
 dev=pd.read_csv(CORE/'reviewer_statistics/tables/predictions_development.csv')
 for cohort in ['development','external_test']:
  pred=pd.read_csv(CORE/f'reviewer_statistics/tables/predictions_{cohort}.csv');g=pred.merge(meta.drop(columns='event'),on='ID',validate='one_to_one') if False else pred.merge(meta[meta.cohort==cohort].drop(columns='event'),on='ID',validate='one_to_one')
  for f in TECH:
   g[f]=pd.to_numeric(g[f],errors='coerce');invalid=g[f].notna()&((g[f]<=0)|~np.isfinite(g[f]));bad=int(invalid.sum());g.loc[invalid,f]=np.nan
   missing.append({'cohort':cohort,'factor':f,'expected_patients':len(pred),'metadata_patients':len(g),'valid_n':int(g[f].notna().sum()),'missing_n':int(g[f].isna().sum()),'invalid_nonpositive_or_nonfinite_n':bad,'unique':g[f].nunique(),'median':g[f].median(),'Q1':g[f].quantile(.25),'Q3':g[f].quantile(.75),'min':g[f].min(),'max':g[f].max()})
  for outcome in ['full_score','texture_score']:
   mu=dev[outcome].mean();sd=dev[outcome].std(ddof=1);g[outcome+'_z']=(g[outcome]-mu)/sd;scales.append({'cohort':cohort,'outcome':outcome,'development_mean':mu,'development_SD':sd})
  matched.append(g)
  eligible=[]
  for f in TECH:
   if g[f].notna().mean()<.8:drops.append({'cohort':cohort,'factor':f,'reason':'less than 80% coverage'});continue
   if g[f].nunique()<2:drops.append({'cohort':cohort,'factor':f,'reason':'constant'});continue
   eligible.append(f)
  j=g.dropna(subset=eligible).copy();keep=[]
  for f in eligible:
   correlated=[a for a in keep if abs(j[[a,f]].corr().iloc[0,1])>=.95]
   if correlated:drops.append({'cohort':cohort,'factor':f,'reason':'absolute Pearson r >=0.95 with earlier factor '+','.join(correlated)})
   else:keep.append(f)
  # Complete-case set is held fixed during prespecified collinearity screening.
  for outcome in ['full_score','texture_score']:
   for f in TECH:
    sub=g.dropna(subset=[f]).copy()
    if sub[f].nunique()<2 or len(sub)<30:continue
    x=base(sub,cohort)
    if f!='field_strength_t' and sub.field_strength_t.nunique()>1:
     sub=sub.dropna(subset=['field_strength_t']);x=base(sub,cohort);add_factor(x,sub,'field_strength_t')
    unit,meaning=add_factor(x,sub,f);rr,di,_,_=ols(sub[outcome+'_z'],x)
    effect=[r for r in rr if r['term']==f][0];allrows.append({'cohort':cohort,'outcome':outcome,'analysis':'single_factor_adjusted','N':len(sub),'factor':f,'predictor_SD':unit,'effect_unit':meaning,**effect})
    sp=spearmanr(sub[f],sub[outcome]);correlations.append({'cohort':cohort,'outcome':outcome,'factor':f,'N':len(sub),'Spearman_rho':sp.statistic,'P':sp.pvalue})
   x=base(j,cohort);units={f:add_factor(x,j,f) for f in keep};rr,di,beta,cov=ols(j[outcome+'_z'],x);positions=[x.columns.get_loc(f) for f in keep];coef=beta[positions];cc=cov[np.ix_(positions,positions)];wald=float(coef@np.linalg.solve(cc,coef));di.update(cohort=cohort,outcome=outcome,analysis='joint',technical_factors=';'.join(keep),technical_Wald_chi2=wald,technical_df=len(positions),technical_omnibus_P=chi2.sf(wald,len(positions)));diagnostics.append(di)
   for r in rr:
    if r['term'] in keep:
     f=r['term'];allrows.append({'cohort':cohort,'outcome':outcome,'analysis':'joint','N':len(j),'factor':f,'predictor_SD':units[f][0],'effect_unit':units[f][1],**r})
   if outcome=='full_score':
    for f in keep:
     v=x[f].to_numpy();a=x.drop(columns=f).to_numpy();res=v-a@np.linalg.lstsq(a,v,rcond=None)[0];vif=np.sum((v-v.mean())**2)/np.sum(res**2);vifs.append({'cohort':cohort,'factor':f,'VIF':vif,'high_VIF_gt10':vif>10})
  print(cohort,'matched',len(g),'joint',len(j),'factors',keep,flush=True)
 results=pd.DataFrame(allrows)
 for _,ix in results.groupby(['cohort','outcome','analysis']).groups.items():results.loc[ix,'Holm_P']=holm(results.loc[ix,'P'].tolist())
 corr=pd.DataFrame(correlations)
 for _,ix in corr.groupby(['cohort','outcome']).groups.items():corr.loc[ix,'Holm_P']=holm(corr.loc[ix,'P'].tolist())
 sheets={'Associations':results,'Missingness':pd.DataFrame(missing),'Joint_diagnostics':pd.DataFrame(diagnostics),'Excluded_factors':pd.DataFrame(drops),'VIF':pd.DataFrame(vifs),'Spearman':corr,'Score_scaling':pd.DataFrame(scales),'Matched_data':pd.concat(matched,ignore_index=True)}
 # All current protocol CMR candidates; descriptive cross-cohort shift, no model fitting.
 policy=json.loads((CORE/'nested_validation/tabular_full/all_features/protocol.json').read_text());tables={}
 for cohort,suffix in [('development','train'),('external_test','external')]:
  truth=pd.read_csv(CORE/'inputs/analysis_manifest.csv').query('cohort==@cohort')[['ID','event']];data=pd.read_excel(CORE.parent/f'CMR parameter/{suffix}.xlsx');data=data.drop(columns='event').merge(truth,on='ID',validate='one_to_one');tables[cohort]=data
 shifts=[]
 for f in policy['cmr_candidates']:
  a=tables['development'][f].dropna();b=tables['external_test'][f].dropna();den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
  shifts.append({'variable':f,'development_n':len(a),'external_n':len(b),'development_mean':a.mean(),'external_mean':b.mean(),'development_SD':a.std(),'external_SD':b.std(),'development_median':a.median(),'external_median':b.median(),'SMD_external_minus_development':(b.mean()-a.mean())/den if den else np.nan,'note':'Original numeric coding; sex is descriptive coded variable, not continuous clinical measurement' if f=='sex' else 'Signed difference / square root of mean within-cohort variances'})
 sheets['CMR_cohort_shift']=pd.DataFrame(shifts)
 for name,df in sheets.items():df.to_csv(ROOT/f'results/{name}.csv',index=False)
 with pd.ExcelWriter(ROOT/'results/采集因素与新Rscore_完整结果.xlsx') as writer:
  for name,df in sheets.items():df.to_excel(writer,sheet_name=name,index=False)
 paths=[CORE/'reviewer_statistics/tables/predictions_development.csv',CORE/'reviewer_statistics/tables/predictions_external_test.csv',ROOT/'inputs/protocol.json',ROOT/'inputs/extended_acquisition_metadata.csv']
 (ROOT/'audit/source_hashes.json').write_text(json.dumps([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],indent=2))
 (ROOT/'audit/analysis_complete.json').write_text(json.dumps({'complete':True,'association_rows':len(results),'development_metadata_n':1050,'external_metadata_n':309,'joint_complete_cases':pd.DataFrame(diagnostics)[['cohort','N']].drop_duplicates().to_dict('records')},indent=2))
 print(results.query("outcome=='full_score' and analysis=='joint'")[['cohort','factor','coefficient','CI_low','CI_high','Holm_P']].to_string(index=False))
if __name__=='__main__':main()
