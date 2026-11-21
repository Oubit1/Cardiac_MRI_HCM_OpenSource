"""Serial, resumable single-GPU experiment queue with automatic reports."""
from common import ROOT, GPU_UUID, pin_gpu, write_json, status, sha256
pin_gpu()
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from train import train_job, infer_job


def report():
    result = subprocess.run([sys.executable, str(ROOT/'code/report.py')], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with (ROOT/'logs/report.log').open('a') as out:
        out.write(result.stdout)
    if result.returncode:
        raise RuntimeError('Report generation failed; see logs/report.log.')


def config(lr, epochs, seed, mode='full'):
    return {'encoder_lr': float(lr), 'epochs': int(epochs), 'seed': int(seed), 'mode': mode,
            'temporal': 'full', 'microbatch': 8}


def mode_lower(values):
    count = Counter(values)
    return sorted(count, key=lambda x: (-count[x], x))[0]


def run():
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT/'queue.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A queue already holds this experiment lock.')
    # Leave this handle open until the queue exits.
    write_json(ROOT/'queue_process.json', {'pid': os.getpid(), 'gpu_uuid': GPU_UUID})
    protocol = json.loads((ROOT/'plans/protocol.json').read_text())
    while True:
        summary = ROOT/'audit/cache_summary.json'
        if summary.exists():
            cached = json.loads(summary.read_text())
            if cached['patients'] == 1359 and not cached['limited_preflight'] and cached.get('preprocessing_version') == 'float32_before_interpolation_v2':
                break
        status('waiting_for_input_preparation', expected_patients=1359)
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'code/check_preprocessing.py')],cwd=ROOT,check=True)
    # Recheck at the actual start, because preparation may take several minutes.
    idle_observations = 0
    while idle_observations < 3:
        query = subprocess.run(['nvidia-smi',f'--id={GPU_UUID}',
                                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                               capture_output=True,text=True,check=True)
        memory,utilization = [int(x.strip()) for x in query.stdout.strip().split(',')]
        if memory <= 1024 and utilization <= 5:
            idle_observations += 1
        else:
            idle_observations = 0
            status('waiting_for_selected_gpu',gpu_uuid=GPU_UUID,memory_mib=memory,utilization=utilization)
        time.sleep(5)
    if not (ROOT/'audit/submitted_checkpoint_reproduction.json').exists():
        subprocess.run([sys.executable,str(ROOT/'code/audit_submitted_checkpoint.py')],cwd=ROOT,check=True)
    records = json.loads((ROOT/'inputs/prepared_patients.json').read_text())
    dev = np.array([i for i, r in enumerate(records) if r['cohort']=='development'])
    ext = np.array([i for i, r in enumerate(records) if r['cohort']=='external_test'])
    assert len(dev)==1050 and len(ext)==309
    y = np.array([r['event'] for r in records])
    selected = []
    split_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_local, te_local) in enumerate(split_cv.split(dev, y[dev]), 1):
        train, test = dev[tr_local], dev[te_local]
        inner_train, inner_val = train_test_split(train, test_size=.1, stratify=y[train], random_state=42+fold)
        write_json(ROOT/f'plans/fold_{fold}_ids.json',
                   {'outer_train': [records[i]['ID'] for i in train],
                    'inner_train': [records[i]['ID'] for i in inner_train],
                    'inner_validation': [records[i]['ID'] for i in inner_val],
                    'outer_test': [records[i]['ID'] for i in test]})
        candidates = []
        for lr in protocol['training']['encoder_lrs']:
            name = f'inner_full_fold{fold}_lr{lr:g}'
            result = train_job(records, inner_train, inner_val, config(lr,60,42+fold), ROOT/'runs'/name)
            candidates.append({'lr': lr, **result['best']['60']})
            if lr == 1e-5:
                original = result['best']
            report()
        winner = sorted(candidates, key=lambda c: (-c['auc'], c['lr']))[0]
        arms = {'baseline20_full': (1e-5,max(3,original['20']['epoch']),42+fold,'full'),
                'extended60_full': (1e-5,max(3,original['60']['epoch']),42+fold,'full'),
                'tuned60_full': (winner['lr'],max(3,winner['epoch']),42+fold,'full')}
        for arm, (lr, epochs, seed, mode) in arms.items():
            d = ROOT/'runs'/f'oof_{arm}_fold{fold}'
            cfg = config(lr,epochs,seed,mode)
            train_job(records, train, None, cfg, d)
            infer_job(records,test,cfg,d/'final.pt',d/'heldout_predictions.csv')
        # Pair ROI optimization with the LR selected for full images in this fold.
        roi = train_job(records,inner_train,inner_val,config(winner['lr'],60,42+fold,'roi'),
                        ROOT/'runs'/f'inner_roi_fold{fold}')
        roi_epoch = max(3,roi['best']['60']['epoch'])
        cfg = config(winner['lr'],roi_epoch,42+fold,'roi')
        d = ROOT/'runs'/f'oof_tuned60_roi_fold{fold}'
        train_job(records,train,None,cfg,d)
        infer_job(records,test,cfg,d/'final.pt',d/'heldout_predictions.csv')
        for seed in [142,242]:
            cfg = config(winner['lr'],max(3,winner['epoch']),seed+fold)
            d = ROOT/'runs'/f'oof_tuned60_full_seed{seed}_fold{fold}'
            train_job(records,train,None,cfg,d)
            infer_job(records,test,cfg,d/'final.pt',d/'heldout_predictions.csv')
        selected.append({'fold':fold,'encoder_lr':winner['lr'],'tuned_epoch':max(3,winner['epoch']),
                         'baseline_epoch':max(3,original['20']['epoch']),
                         'extended_epoch':max(3,original['60']['epoch']),'roi_epoch':roi_epoch})
        write_json(ROOT/'plans/completed_fold_selections.json', selected)
        report()
    final_lr = mode_lower([s['encoder_lr'] for s in selected])
    final_configs = {
        'baseline20_full': config(1e-5,int(np.median([s['baseline_epoch'] for s in selected])),42),
        'extended60_full': config(1e-5,int(np.median([s['extended_epoch'] for s in selected])),42),
        'tuned60_full': config(final_lr,int(np.median([s['tuned_epoch'] for s in selected])),42),
        'tuned60_roi': config(final_lr,int(np.median([s['roi_epoch'] for s in selected])),42,'roi'),
    }
    for seed in [142,242]:
        final_configs[f'tuned60_full_seed{seed}'] = {**final_configs['tuned60_full'],'seed':seed}
    # Freeze all final configurations before any new external predictions are made.
    write_json(ROOT/'plans/final_locked_configs.json', final_configs)
    report()  # Also writes development-derived operating thresholds.
    for arm, cfg in final_configs.items():
        d = ROOT/'runs'/f'final_{arm}'
        train_job(records,dev,None,cfg,d)
        infer_job(records,ext,cfg,d/'final.pt',d/'external_predictions.csv',features=arm=='tuned60_full')
        if arm=='tuned60_full':
            infer_job(records,dev,cfg,d/'final.pt',d/'development_apparent_features.csv',features=True)
        report()
    # Fixed classifier; no external-based tuning and no new fusion OOF claim.
    subprocess.run([sys.executable,str(ROOT/'code/fusion.py')],cwd=ROOT,check=True)
    write_json(ROOT/'all_training_complete.json', {'completed':True,'protocol_sha256':sha256(ROOT/'plans/protocol.json')})
    report()
    write_json(ROOT/'FINAL_COMPLETE.json', {'completed':True,'response':'response/mViT_response.docx',
               'workbook':'tables/mViT_revision_results.xlsx','protocol_sha256':sha256(ROOT/'plans/protocol.json')})
    status('complete',response='response/mViT_response.docx',workbook='tables/mViT_revision_results.xlsx')


if __name__=='__main__':
    try:
        if json.loads((ROOT/'plans/protocol.json').read_text()).get('active_stage') == 'core24h':
            from core_queue import run as run_core
            run_core()
        else:
            run()
    except Exception as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
