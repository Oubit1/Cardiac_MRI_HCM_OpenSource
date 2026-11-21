"""Tiny real-input software test; outputs are not scientific experiment results."""
from common import ROOT,write_json
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch


def validate(accelerated=False):
    from train import train_job,infer_job
    manifest=pd.read_csv(ROOT/'inputs/analysis_manifest.csv')
    records=[]
    for event in [0,1]:
        count=0
        for row in manifest[(manifest.cohort=='development')&(manifest.event==event)].itertuples():
            files=sorted((ROOT/'cache/metadata').glob(f'development_{row.ID:04d}_slice*.json'))
            if len(files)<2:continue
            meta=[json.loads(p.read_text()) for p in files[:2]]
            if any(m.get('preprocessing_version')!='float32_before_interpolation_v2' for m in meta):continue
            records.append({'ID':int(row.ID),'event':int(event),'cohort':'development',
                            'full_paths':[m['full_cache'] for m in meta],
                            'roi_paths':[m['roi_cache'] for m in meta]})
            count+=1
            if count==2:break
        if count!=2:raise RuntimeError('Insufficient prepared cases for the small software test.')
    directory=ROOT/('audit/software_accelerated_pipeline_smoke' if accelerated else 'audit/software_pipeline_smoke')
    cfg={'encoder_lr':1e-5,'epochs':3,'seed':991,'mode':'full','temporal':'full','microbatch':8}
    train_job(records,[0,2],[1,3],cfg,directory)
    infer_job(records,[1,3],cfg,directory/'best_60.pt',directory/'software_only_predictions.csv')
    # Exercise resume/loading of model, optimizer, scaler and RNG without an extra epoch.
    completed=directory/'complete.json'
    if not (directory/'first_completion.json').exists():completed.rename(directory/'first_completion.json')
    train_job(records,[0,2],[1,3],cfg,directory)
    frame=pd.read_csv(directory/'software_only_predictions.csv')
    assert len(frame)==2 and frame.probability.between(0,1).all()
    if accelerated:
        source=ROOT/'audit/software_accelerated_refit'
        destination=ROOT/'audit/software_accelerated_reused_refit'
        refit_cfg={**cfg,'epochs':1}
        train_job(records,[0,1,2,3],None,refit_cfg,source)
        reused=train_job(records,[0,1,2,3],None,refit_cfg,destination)
        assert reused['reused_from']==str(source.relative_to(ROOT))
        assert (source/'final.pt').samefile(destination/'final.pt')
    write_json(ROOT/('audit/accelerated_pipeline_validation.json' if accelerated else 'audit/pipeline_validation.json'),{'passed':True,
        'checks':['real cached input loading','head warmup','encoder unfreezing','mixed-precision backward',
                  'optimizer update','per-epoch evaluation','atomic checkpoint','checkpoint/RNG resume',
                  'checkpoint inference','report refresh'],
        'role':'Software only; four cases/two slices, not scientific evidence or hyperparameter selection'})
    print('End-to-end software validation passed.')


if __name__=='__main__':validate()
