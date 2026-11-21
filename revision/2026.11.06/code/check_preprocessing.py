"""Validate cached transforms against the archived cine preprocessing code."""
from common import ROOT,write_json
import argparse
import json
import sys
import numpy as np
import torch
sys.path.insert(0,str(ROOT/'code/vendor'))
from legacy_trainer import prepare_cine
from train import Patients


def main():
    p=argparse.ArgumentParser();p.add_argument('--sample-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    eligible=[]
    for path in sorted((ROOT/'cache/metadata').glob('*.json')):
        value=json.loads(path.read_text())
        if value.get('preprocessing_version')=='float32_before_interpolation_v2':eligible.append((path,value))
    if not args.sample_only and len(eligible)!=10855:
        raise ValueError(f'Expected 10855 selected cached slices, found {len(eligible)}.')
    if not eligible:raise ValueError('No current cache entries.')
    chosen=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(12,len(eligible))).round().astype(int))]
    checks=[]
    for path,m in chosen:
        record={'ID':int(m['ID']),'event':0,'full_paths':[m['full_cache']],'roi_paths':[m['roi_cache']]}
        for augment in [False,True]:
            torch.manual_seed(555)
            reference=prepare_cine(m['current_path'],augment=augment,temporal_mode='full')
            torch.manual_seed(555)
            actual=Patients([record],[0],augment=augment)[0][0][0]
            error=float((reference-actual).abs().max())
            if error>1e-6:raise AssertionError(f'{path.name}: max error {error}')
            checks.append({'sample':path.name,'augmentation':augment,'max_abs_error':error})
    write_json(ROOT/'audit/cache_transform_equivalence.json',{'complete_cache':not args.sample_only,
               'eligible_slices':len(eligible),'checks':checks,'tolerance':1e-6,'passed':True})
    print(f'Passed {len(checks)} cached/legacy transform comparisons.')


if __name__=='__main__':main()
