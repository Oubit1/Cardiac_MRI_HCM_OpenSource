"""User-prioritized core supplement: five-fold duration and matched ROI comparisons."""
from common import ROOT, GPU_UUID, write_json, status, sha256, timestamp
from train import train_job, infer_job
from run_queue import config, report
import fcntl
import json
import os
import subprocess
import sys
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def run():
    lock=(ROOT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(ROOT/'queue_process.json',{'pid':os.getpid(),'gpu_uuid':GPU_UUID,'stage':'core24h'})
    cached=json.loads((ROOT/'audit/cache_summary.json').read_text())
    assert cached['patients']==1359 and not cached['limited_preflight']
    assert cached['preprocessing_version']=='float32_before_interpolation_v2'
    subprocess.run([sys.executable,str(ROOT/'code/check_preprocessing.py')],cwd=ROOT,check=True)
    idle=0
    while idle<3:
        q=subprocess.run(['nvidia-smi',f'--id={GPU_UUID}','--query-gpu=memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        memory,utilization=map(int,q.stdout.strip().split(','))
        idle=idle+1 if memory<=1024 and utilization<=5 else 0
        if not idle:status('waiting_for_selected_gpu',gpu_uuid=GPU_UUID,memory_mib=memory,utilization=utilization)
        time.sleep(5)
    records=json.loads((ROOT/'inputs/prepared_patients.json').read_text())
    dev=np.array([i for i,r in enumerate(records) if r['cohort']=='development'])
    ext=np.array([i for i,r in enumerate(records) if r['cohort']=='external_test'])
    assert len(dev)==1050 and len(ext)==309
    y=np.array([r['event'] for r in records])
    selected=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(dev,y[dev]),1):
        train,test=dev[tr],dev[te]
        inner_train,inner_val=train_test_split(train,test_size=.1,stratify=y[train],random_state=42+fold)
        write_json(ROOT/f'plans/fold_{fold}_ids.json',{'outer_train':[records[i]['ID'] for i in train],
                   'inner_train':[records[i]['ID'] for i in inner_train],
                   'inner_validation':[records[i]['ID'] for i in inner_val],
                   'outer_test':[records[i]['ID'] for i in test]})
        result=train_job(records,inner_train,inner_val,config(1e-5,60,42+fold),
                         ROOT/'runs'/f'inner_full_fold{fold}_lr1e-05')
        epochs20=max(3,result['best']['20']['epoch'])
        epochs60=max(3,result['best']['60']['epoch'])
        for arm,epochs,mode in [('baseline20_full',epochs20,'full'),('extended60_full',epochs60,'full'),
                               ('roi_matched_full',epochs60,'roi')]:
            cfg=config(1e-5,epochs,42+fold,mode)
            directory=ROOT/'runs'/f'oof_{arm}_fold{fold}'
            train_job(records,train,None,cfg,directory)
            infer_job(records,test,cfg,directory/'final.pt',directory/'heldout_predictions.csv')
        selected.append({'fold':fold,'encoder_lr':1e-5,'baseline_epoch':epochs20,
                         'extended_epoch':epochs60,'roi_epoch':epochs60,
                         'roi_epoch_policy':'matched full-image duration; no independent tuning'})
        write_json(ROOT/'plans/completed_fold_selections.json',selected)
        report()
    final_configs={
        'baseline20_full':config(1e-5,int(np.median([s['baseline_epoch'] for s in selected])),42),
        'extended60_full':config(1e-5,int(np.median([s['extended_epoch'] for s in selected])),42),
        'roi_matched_full':config(1e-5,int(np.median([s['extended_epoch'] for s in selected])),42,'roi')}
    write_json(ROOT/'plans/final_locked_configs.json',final_configs)
    report()
    for arm,cfg in final_configs.items():
        directory=ROOT/'runs'/f'final_{arm}'
        train_job(records,dev,None,cfg,directory)
        infer_job(records,ext,cfg,directory/'final.pt',directory/'external_predictions.csv',features=arm=='extended60_full')
        if arm=='extended60_full':
            infer_job(records,dev,cfg,directory/'final.pt',directory/'development_apparent_features.csv',features=True)
        report()
    subprocess.run([sys.executable,str(ROOT/'code/fusion.py')],cwd=ROOT,check=True)
    protocol=json.loads((ROOT/'plans/protocol.json').read_text())
    completion={'completed':True,'scope':'amended core24h protocol','finished_at':timestamp(),
                'protocol_sha256':sha256(ROOT/'plans/protocol.json'),
                'deferred_experiments':protocol['deferred_experiments']}
    write_json(ROOT/'all_training_complete.json',completion)
    report()
    write_json(ROOT/'FINAL_COMPLETE.json',{**completion,'response':'response/mViT_response.docx',
                                         'workbook':'tables/mViT_revision_results.xlsx'})
    status('core_complete',**completion,response='response/mViT_response.docx',workbook='tables/mViT_revision_results.xlsx')
