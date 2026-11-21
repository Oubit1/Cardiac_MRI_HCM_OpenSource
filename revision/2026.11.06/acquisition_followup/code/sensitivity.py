from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(ROOT/'code'))
from analyze import base,add_factor,ols
sys.path.insert(0,str(CORE/'code'))
from revision_statistics import paired,holm

def main():
 g=pd.read_csv(ROOT/'results/Matched_data.csv');diag=pd.read_csv(ROOT/'results/Joint_diagnostics.csv');rows=[];drop=[];diagnostics=[]
 for cohort,sub in g.groupby('cohort'):
  terms=diag.query('cohort==@cohort').iloc[0].technical_factors.split(';');sub=sub.dropna(subset=terms);x=base(sub,cohort)
  for f in terms:add_factor(x,sub,f)
  while True:
   vifs={}
   for f in terms:
    v=x[f].to_numpy();other=x.drop(columns=f).to_numpy();res=v-other@np.linalg.lstsq(other,v,rcond=None)[0];vifs[f]=np.sum((v-v.mean())**2)/np.sum(res**2)
   chosen=max(vifs,key=vifs.get)
   if vifs[chosen]<=10:break
   drop.append({'cohort':cohort,'factor':chosen,'VIF_at_removal':vifs[chosen],'step':len([r for r in drop if r['cohort']==cohort])+1});terms.remove(chosen);x=x.drop(columns=chosen)
  for outcome in ['full_score','texture_score']:
   rr,dd,_,_=ols(sub[outcome+'_z'],x);dd.update(cohort=cohort,outcome=outcome,technical_factors=';'.join(terms));diagnostics.append(dd)
   for r in rr:
    if r['term'] in terms:rows.append({'cohort':cohort,'outcome':outcome,'N':len(sub),'final_VIF':vifs[r['term']],**r})
 out=pd.DataFrame(rows)
 for _,ix in out.groupby(['cohort','outcome']).groups.items():out.loc[ix,'Holm_P']=holm(out.loc[ix,'P'].tolist())
 out.to_csv(ROOT/'results/VIF_sensitivity.csv',index=False);pd.DataFrame(drop).to_csv(ROOT/'results/VIF_removals.csv',index=False);pd.DataFrame(diagnostics).to_csv(ROOT/'results/VIF_sensitivity_diagnostics.csv',index=False)
 (ROOT/'inputs/sensitivity_addendum.json').write_text(json.dumps({'reason':'Main joint models had VIF>10; added diagnostic sensitivity without selecting on Rscore associations','rule':'Same joint complete cases; iteratively remove technical term with maximum VIF until all <=10; never remove event or fold indicators; primary model unchanged','scope':'Exploratory; dropping field strength can reduce confounding control, so this does not replace the original field-adjusted results'},indent=2))
 # Archived predictions, identical current external labels; no patient names loaded.
 new=pd.read_csv(CORE/'reviewer_statistics/tables/predictions_external_test.csv')
 old=pd.read_csv(CORE.parent/'Data/Model construction/CatBoost/prediction_results_external.csv',usecols=['ID','true_label','predicted_probability','predicted_probability_without_Rscore']).rename(columns={'true_label':'archived_catboost_event','predicted_probability':'archived_combined','predicted_probability_without_Rscore':'archived_CMR'})
 rad=pd.read_csv(CORE.parent/'Data/Radiomics/Rscore_OOF_results/Rscore_external_predictions.csv',usecols=['ID','event','Probability_external']).rename(columns={'event':'archived_rscore_event','Probability_external':'archived_Rscore'})
 d=new[['ID','event','CMR','Rscore_full','CMR_Rscore_full']].merge(old,on='ID',validate='one_to_one').merge(rad,on='ID',validate='one_to_one');assert len(d)==309 and d.ID.is_unique
 comps=[]
 for label,a,b,old_y in [('CMR','archived_CMR','CMR','archived_catboost_event'),('Rscore','archived_Rscore','Rscore_full','archived_rscore_event'),('CMR + Rscore','archived_combined','CMR_Rscore_full','archived_catboost_event')]:
  comps.append({'model':label,'N':len(d),'current_events':int(d.event.sum()),'archived_events':int(d[old_y].sum()),'label_disagreements':int((d.event!=d[old_y]).sum()),'archived_probability_AUC_original_labels':roc_auc_score(d[old_y],d[a]),'archived_probability_AUC_current_labels':roc_auc_score(d.event,d[a]),'new_probability_AUC_current_labels':roc_auc_score(d.event,d[b]),**paired(d.event,d[a],d[b],rounds=2000)})
 comps=pd.DataFrame(comps);comps['Holm_P']=holm(comps.DeLong_P.tolist());comps.to_csv(ROOT/'results/Archived_new_same_label_comparison.csv',index=False);d.to_csv(ROOT/'audit/Archived_new_same_patient_predictions.csv',index=False)
 d[(d.event!=d.archived_catboost_event)|(d.event!=d.archived_rscore_event)][['ID','event','archived_catboost_event','archived_rscore_event']].to_csv(ROOT/'audit/Archived_label_differences.csv',index=False)
 print(comps[['model','archived_probability_AUC_current_labels','new_probability_AUC_current_labels','Delta_AUC_B_minus_A','Delta_CI_low','Delta_CI_high']].to_string(index=False))
if __name__=='__main__':main()
