from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(CORE/'code'));from revision_statistics import threshold
D=pd.read_csv(ROOT/'tables/predictions_development.csv');cut=threshold(D.event,D.CMR_Rscore_full)
rows=[];counts=[]
for cohort in ['development','external_test']:
 g=pd.read_csv(ROOT/f'tables/predictions_{cohort}.csv');y=g.event.to_numpy();a=(g.CMR.to_numpy()>=cut).astype(int);b=(g.CMR_Rscore_full.to_numpy()>=cut).astype(int);change=b-a
 pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
 event=change[pos].mean();nonevent=-change[neg].mean();rng=np.random.default_rng(42);boot=[]
 for _ in range(2000):
  ev=change[rng.choice(pos,len(pos))].mean();ne=-change[rng.choice(neg,len(neg))].mean();boot.append([ev,ne,ev+ne])
 boot=np.array(boot)
 for j,(name,v) in enumerate([('NRI_event',event),('NRI_nonevent',nonevent),('NRI_total',event+nonevent)]):
  lo,hi=np.quantile(boot[:,j],[.025,.975]);rows.append(dict(cohort=cohort,A='CMR',B='CMR_Rscore_full',metric=name,estimate=v,CI_low=lo,CI_high=hi,threshold_A=cut,threshold_B=cut,bootstrap=2000))
 for e in [0,1]:
  for x in [0,1]:
   for z in [0,1]:counts.append(dict(cohort=cohort,event=e,CMR_category=x,combined_category=z,n=int(((y==e)&(a==x)&(b==z)).sum())))
 # Binary categorical NRI identities under a common boundary.
 assert np.isclose(event,b[pos].mean()-a[pos].mean())
 assert np.isclose(nonevent,(1-b[neg]).mean()-(1-a[neg]).mean())
result=pd.DataFrame(rows);result.to_csv(ROOT/'tables/NRI_categorical.csv',index=False);pd.DataFrame(counts).to_csv(ROOT/'tables/NRI_categorical_counts.csv',index=False)
(ROOT/'audit/categorical_nri_protocol.json').write_text(json.dumps({'threshold':cut,'rule':'Existing final combined-model development OOF sensitivity >=0.95 threshold applied identically to both models','analysis':'supplementary binary categorical NRI, not prospectively prespecified clinical risk categories','bootstrap':'2000 paired stratified patient resamples; threshold fixed','external_threshold_tuning':False,'identities_checked':True},indent=2))
print(result.to_string(index=False))
