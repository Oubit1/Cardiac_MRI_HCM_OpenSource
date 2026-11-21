"""Batched inference of existing fold models; never performs an optimizer update."""
from settings import ROOT,CORE,write_json,sha,now
import argparse,json,os,sys,time,gc
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,required=True)
    parser.add_argument('--gpu',required=True,help='GPU UUID resolved by the controller')
    parser.add_argument('--patient-batch',default='auto')
    parser.add_argument('--slice-microbatch',type=int,default=32)
    parser.add_argument('--loader-workers',type=int,default=4)
    parser.add_argument('--benchmark-only',action='store_true')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    sys.path.insert(0,str(CORE/'code'))
    import common
    common.GPU_UUID=args.gpu
    import torch,numpy as np,pandas as pd
    from train import make_model,seed_all,Patients
    from torch.utils.data import DataLoader
    directory=ROOT/'runs'/f'fold_{args.fold}'
    split=json.loads((directory/'split.json').read_text())
    weight=Path(split['image_checkpoint'])
    assert sha(weight)==split['image_checkpoint_sha256']
    cfg=json.loads((weight.parent/'config.json').read_text())
    assert cfg['train_ids']==split['outer_train'] and not cfg['validation_ids']
    records=json.loads((CORE/'inputs/prepared_patients.json').read_text())
    by_id={r['ID']:i for i,r in enumerate(records) if r['cohort']=='development'}
    train_ids=split['outer_train'];test_ids=split['outer_test']
    seed_all(cfg['seed'])
    assert torch.cuda.device_count()==1
    model=make_model(args.slice_microbatch)
    payload=torch.load(weight,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True);del payload
    model.eval()
    @torch.inference_mode()
    def forward(items,microbatch=None):
        sizes=[len(x[0]) for x in items]
        videos=torch.cat([x[0] for x in items]).cuda(non_blocking=True)
        old=model.encoder_microbatch
        if microbatch is not None:model.encoder_microbatch=microbatch
        try:
            with torch.autocast('cuda',dtype=torch.float16):
                encoded=model.encode(videos);features=[];probabilities=[];start=0
                for size in sizes:
                    z,_=model.aggregator(encoded[start:start+size]);start+=size
                    logit=model.classifier(z).squeeze(0)
                    features.append(z.float().cpu().numpy())
                    probabilities.append(float(torch.sigmoid(logit).cpu()))
            return np.stack(features),np.array(probabilities)
        finally:model.encoder_microbatch=old
    # Tuning uses only training inputs and speed/memory, never heldout outcomes or AUC.
    representative=max(train_ids,key=lambda ident:len(records[by_id[ident]]['full_paths']))
    sample=Patients(records,[by_id[representative]],'full',False)[0]
    batches=[1,2,4,8] if args.patient_batch=='auto' else [int(args.patient_batch)]
    measurements=[]
    for batch in batches:
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            forward([sample]*batch);torch.cuda.synchronize()
            timings=[]
            for _ in range(2):
                began=time.monotonic();forward([sample]*batch);torch.cuda.synchronize();timings.append(time.monotonic()-began)
            measurements.append({'batch':batch,'seconds_per_patient':float(np.median(timings))/batch,
                                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache()
    if not measurements:raise RuntimeError('No candidate batch fits GPU memory; reduce --slice-microbatch.')
    batch=min(measurements,key=lambda x:x['seconds_per_patient'])['batch']
    check_ids=train_ids[:2]
    check=[Patients(records,[by_id[x]],'full',False)[0] for x in check_ids]
    sequential=[forward([x],8) for x in check]
    ff,pp=forward(check)
    ref=np.concatenate([x[0] for x in sequential]);refp=np.concatenate([x[1] for x in sequential])
    relative=float(np.linalg.norm(ff-ref)/max(np.linalg.norm(ref),1e-12))
    maximum=float(np.max(np.abs(pp-refp)))
    assert relative<.01 and maximum<.005,(relative,maximum)
    receipt={'time':now(),'gpu_uuid':args.gpu,'patient_batch':batch,'slice_microbatch':args.slice_microbatch,
             'code_sha256':sha(__file__),
             'timings':measurements,'feature_relative_l2_vs_sequential':relative,'probability_max_delta_vs_sequential':maximum,
             'checkpoint_sha256':sha(weight),'purpose':'Inference only; AMP numerical tolerance, not bitwise identity'}
    write_json(directory/('gpu_benchmark.json' if args.benchmark_only else 'feature_execution.json'),receipt)
    print(json.dumps(receipt),flush=True)
    if args.benchmark_only:return
    began=time.monotonic()
    for role,ids in [('train',train_ids),('heldout',test_ids)]:
        path=directory/f'image_{role}.npz'
        dataset=Patients(records,[by_id[x] for x in ids],'full',False)
        loader=DataLoader(dataset,batch_size=batch,shuffle=False,collate_fn=list,
                          num_workers=args.loader_workers,pin_memory=True,
                          **({'prefetch_factor':2} if args.loader_workers else {}))
        vectors=[];probs=[];output_ids=[]
        for step,items in enumerate(loader):
            f,p=forward(items);vectors.append(f);probs.append(p);output_ids.extend(int(x[2]) for x in items)
            if step%20==0:print(f'fold={args.fold} role={role} patients={len(output_ids)}/{len(ids)}',flush=True)
        assert output_ids==ids
        vectors=np.concatenate(vectors);probs=np.concatenate(probs)
        assert vectors.shape==(len(ids),256) and np.isfinite(vectors).all()
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as out:np.savez_compressed(out,ID=output_ids,features=vectors,probability=probs)
        tmp.replace(path)
        if role=='heldout':
            old=pd.read_csv(weight.parent/'heldout_predictions.csv').set_index('ID').loc[ids]
            delta=float(np.max(np.abs(old.probability.to_numpy()-probs)))
            assert delta<.005,delta
            receipt['heldout_probability_max_delta_from_core']=delta
    receipt['seconds']=time.monotonic()-began
    receipt['completed']=True
    write_json(directory/'features_complete.json',receipt)

if __name__=='__main__':main()
