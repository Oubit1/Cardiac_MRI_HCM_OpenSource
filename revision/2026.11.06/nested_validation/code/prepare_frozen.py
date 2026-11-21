"""Reuse only outcome-independent frozen embeddings; align by cohort and patient ID."""
from settings import *
import numpy as np
import pandas as pd
SOURCE=PROJECT/'experiments/hcm_foundation_benchmark/results/foundation_embeddings_nifti_v1/patient_embeddings.npz'

def attach(directory,train_ids,test_ids,external=False):
    with np.load(SOURCE,allow_pickle=True) as z:
        features=z['embeddings'];cohorts=z['cohort'];ids=z['patient_id']
    assert features.shape[1]==512 and np.isfinite(features).all()
    for role,wanted,cohort in [('train',train_ids,'development'),('heldout',test_ids,'external_test' if external else 'development')]:
        mask=cohorts==cohort;frame=pd.DataFrame(features[mask],index=ids[mask])
        assert not frame.index.duplicated().any() and set(wanted)<=set(frame.index)
        values=frame.loc[wanted].to_numpy()
        assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4)
        destination=Path(directory)/f'frozen_{role}.npz'
        if destination.exists():
            with np.load(destination) as old:
                if np.array_equal(old['ID'],np.asarray(wanted)) and np.array_equal(old['features'],values):continue
            raise RuntimeError(f'Frozen features changed: {destination}')
        temp=destination.with_suffix('.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,ID=np.asarray(wanted),features=values)
        temp.replace(destination)

def main():
    metadata=ROOT/'audit/frozen_source.json'
    if metadata.exists() and json.loads(metadata.read_text())['sha256']!=sha(SOURCE):
        raise RuntimeError('Frozen source changed since preparation')
    for i in range(1,6):
        directory=ROOT/f'runs/fold_{i}';spec=json.loads((directory/'split.json').read_text())
        attach(directory,spec['outer_train'],spec['outer_test'])
    write_json(ROOT/'audit/frozen_source.json',{'path':str(SOURCE),'sha256':sha(SOURCE),'labels_used':False,'representation':'512D mean pooling then L2 normalization'})
    print('Frozen features aligned for all five matched folds.')

if __name__=='__main__':main()
