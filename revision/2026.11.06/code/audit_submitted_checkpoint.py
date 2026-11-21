"""Reproduce archived submitted image-only probabilities; never selects new models."""
from common import ROOT,write_json,sha256
import json
import numpy as np
import pandas as pd
import torch
from train import make_model,predict,seed_all
from revision_statistics import auc


def main():
    records=json.loads((ROOT/'inputs/prepared_patients.json').read_text())
    indices=[i for i,r in enumerate(records) if r['cohort']=='external_test']
    source=ROOT/'inputs/submitted_candidate_model.pt'
    seed_all(42)
    model=make_model()
    saved=torch.load(source,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model_state'],strict=True)
    cfg={'seed':42,'mode':'full','temporal':'full'}
    frame,_=predict(model,records,indices,cfg)
    frame.to_csv(ROOT/'audit/reproduced_submitted_external_predictions.csv',index=False)
    archived=pd.read_csv(ROOT/'inputs/submitted_ft_external_test_predictions.csv')
    merged=frame.merge(archived[['ID','event','probability']],on='ID',validate='one_to_one',suffixes=('_reproduced','_submitted'))
    assert len(merged)==309 and (merged.event_reproduced==merged.event_submitted).all()
    delta=np.abs(merged.probability_reproduced-merged.probability_submitted)
    write_json(ROOT/'audit/submitted_checkpoint_reproduction.json',{
        'checkpoint':str(source),'checkpoint_sha256':sha256(source),'locked_epochs':int(saved['locked_epochs']),
        'threshold':float(saved['threshold']),'n':len(merged),'max_abs_probability_difference':float(delta.max()),
        'mean_abs_probability_difference':float(delta.mean()),'tolerance':.001,
        'reproduced_auc':auc(merged.event_reproduced,merged.probability_reproduced),
        'submitted_auc':auc(merged.event_submitted,merged.probability_submitted),
        'matches_within_tolerance':bool(delta.max()<=.001),
        'role':'Archived checkpoint provenance only; not new model selection or an untouched test'})


if __name__=='__main__':main()
