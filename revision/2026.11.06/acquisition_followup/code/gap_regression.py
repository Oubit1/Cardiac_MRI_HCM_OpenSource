from pathlib import Path
import sys,json
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
sys.path.insert(0,str(ROOT/'code'));from analyze import ols,base,add_factor
sys.path.insert(0,str(CORE/'code'));from revision_statistics import holm
tech=json.loads((ROOT/'inputs/protocol.json').read_text())['technical_factors'];tech=['derived_gap_mm' if k=='spacing_between_slices_mm' else k for k in tech]
all_data=pd.read_csv(ROOT/'results/Matched_data.csv');rows=[];audit=[]
for cohort in ['development','external_test']:
 g=all_data[all_data.cohort==cohort].copy();factors=[k for k in tech if g[k].nunique()>1];g=g.dropna(subset=factors+['full_score_z']);x=base(g,cohort)
 for k in factors:add_factor(x,g,k)
 result,diag,_,_=ols(g.full_score_z,x)
 sub=pd.DataFrame(result);sub=sub[sub.term.isin(factors)].copy();sub['Holm_P']=holm(sub.P);sub['cohort']=cohort;sub['N']=len(g);rows.append(sub)
 old=pd.read_csv(ROOT/'results/Joint_diagnostics.csv');old=old[(old.cohort==cohort)&(old.outcome=='full_score')].iloc[0]
 assert np.isclose(diag['R2'],old.R2,atol=1e-10)
 audit.append(dict(cohort=cohort,N=len(g),negative_derived_gap_n=int((g.derived_gap_mm<0).sum()),R2=diag['R2'],same_fitted_space_as_spacing_plus_thickness=True))
result=pd.concat(rows);result.to_csv(ROOT/'results/Gap_joint_associations.csv',index=False)
(ROOT/'audit/gap_regression.json').write_text(json.dumps({'definition':'DICOM SpacingBetweenSlices - SliceThickness, signed derived gap; not independently verified prescribed gap','negative_values':'retained as signed values (may denote overlap or header discrepancy), not imputed or clipped','model':'same joint factor model, replacing center-to-center spacing with derived gap, retaining thickness','validation':audit},indent=2))
print(result[result.term=='derived_gap_mm'].to_string(index=False))
