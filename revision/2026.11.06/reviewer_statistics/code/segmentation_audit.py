import os
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('JMRI_SEGMENTATION_ROOT', 'Automated_Annotation_of_Cine'))
split=json.loads((SOURCE/'manifests/train_val_patient_split_seed42.json').read_text())
for c,sets in split['patient_lists'].items():assert not set(sets['train'])&set(sets['val'])
rows=[];perpatient=[];sources=[]
for run in ['unet2d_myo_all_centers','reproduce_cmr']:
    r=SOURCE/'runs'/run;weight=r/'weights/best_model.pt'
    checkpoint=torch.load(weight,map_location='cpu',weights_only=False);epoch=int(checkpoint['epoch']);del checkpoint
    hist=pd.read_csv(r/'history.csv');assert epoch==int(hist.loc[hist.val_dice.idxmax(),'epoch'])
    metrics_path=r/f'validation/epoch_{epoch:03d}/metrics.json';metrics=json.loads(metrics_path.read_text())
    volumes=pd.read_csv(r/f'validation/epoch_{epoch:03d}/per_volume_metrics.csv',dtype={'patient':str})
    for c,sub in volumes.groupby('center'):assert set(sub.patient)==set(split['patient_lists'][c]['val'])
    patients=volumes.groupby(['center','patient'])[['dice','iou']].mean().reset_index()
    patients['patient_hash']=[hashlib.sha256((c+'/'+p).encode()).hexdigest()[:12] for c,p in zip(patients.center,patients.patient)]
    perpatient.append(patients.drop(columns='patient').assign(run=run))
    for center in ['Overall',*sorted(split['patient_lists'])]:
        p=patients if center=='Overall' else patients[patients.center==center]
        reported=metrics if center=='Overall' else metrics['by_center'][center]
        rng=np.random.default_rng(42);values=p.dice.to_numpy();boots=np.array([rng.choice(values,len(values)).mean() for _ in range(2000)])
        lo,hi=np.quantile(boots,[.025,.975])
        rows.append({'run':run,'best_epoch':epoch,'center':center,'patients':len(p),'volumes':reported['volumes'],'frames':reported['frames'],
             'pooled_pixel_Dice':reported['dice'],'IoU':reported['iou'],'precision':reported['precision'],'recall':reported['recall'],
             'patient_mean_volume_Dice':values.mean(),'patient_Dice_CI_low':lo,'patient_Dice_CI_high':hi,'role':'model-selection validation; not independent test'})
    sources.append({'run':run,'checkpoint':str(weight),'sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'metrics':str(metrics_path),'best_epoch':epoch})
pd.DataFrame(rows).to_csv(ROOT/'tables/segmentation_validation.csv',index=False)
pd.concat(perpatient).to_csv(ROOT/'audit/segmentation_patient_aggregate.csv',index=False)
(ROOT/'audit/segmentation_provenance.json').write_text(json.dumps({'source':str(SOURCE),'sources':sources,'split_summary':split['summary'],
    'split_integrity_passed':True,'independent_test_found':False,'current_ROI_checkpoint_link_confirmed':False,
    'HCM_patient_overlap':'not established; numeric anonymized IDs are not cross-dataset identity keys',
    'metric_definitions':'Source Dice pools pixels across annotated frames; patient mean is equal-weight mean of per-volume Dice within each patient then across patients. Bootstrap CI does not account for best-epoch selection.'},ensure_ascii=False,indent=2))
print('Segmentation records and validation split audited.')
