"""Prespecified external-only representation-fusion sensitivity, no OOF claims."""
from common import ROOT, write_json
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def main():
    protocol=json.loads((ROOT/'plans/protocol.json').read_text())
    source=protocol.get('fusion_source_arm','tuned60_full')
    directory=ROOT/'runs'/f'final_{source}'
    dev=np.load(directory/'development_apparent_features.npz')
    ext=np.load(directory/'external_predictions.npz')
    manifest=pd.read_csv(ROOT/'inputs/analysis_manifest.csv')
    clinical=['MWT','LAs','LVGRS','LVGLS']
    a=manifest[manifest.cohort=='development'].set_index('ID').loc[dev['ID']]
    b=manifest[manifest.cohort=='external_test'].set_index('ID').loc[ext['ID']]
    output=pd.DataFrame({'ID':b.index,'event':b.event.to_numpy()})
    classifiers={}
    for name,cols,image in [('CMR_fixed',clinical,False),('Rscore_CMR_fixed',['Rscore',*clinical],False),
                            ('mViT_CMR_fixed',clinical,True),('mViT_Rscore_CMR_fixed',['Rscore',*clinical],True)]:
        x=a[cols].to_numpy(float); z=b[cols].to_numpy(float)
        if image:
            x=np.c_[dev['features'],x];z=np.c_[ext['features'],z]
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                            LogisticRegression(penalty='elasticnet',solver='saga',C=.1,l1_ratio=.5,
                                               max_iter=10000,tol=1e-5,random_state=42))
        model.fit(x,a.event)
        output[name]=model.predict_proba(z)[:,1]
        classifiers[name]=model
    output.to_csv(directory/'external_fixed_fusion_predictions.csv',index=False)
    joblib.dump(classifiers,directory/'fixed_fusion_models.joblib')
    write_json(directory/'fusion_policy.json',{'role':'External-only post hoc sensitivity','image_source_arm':source,
                'classifier':'C=0.1, l1_ratio=0.5 fixed by protocol before new training',
                'development_features':'Apparent supervised representations; not OOF',
                'no_development_performance_claim':True,'no_new_fusion_threshold':True,
                'cmr_selection':'Historical four variables from full-development selection; not claimed nested',
                'rscore_source':'User-confirmed 26.9.19 scripts; values in analysis_manifest.csv',
                'external_reuse':'Previously examined cohort; not fresh independent confirmation'})


if __name__=='__main__':main()
