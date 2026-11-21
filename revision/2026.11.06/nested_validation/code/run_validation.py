"""Single entry point: prepare/check/test, selectable stages, multi-GPU inference, CPU folds."""
from settings import *
import argparse,subprocess,sys,fcntl,queue,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed

LOCKS={}

def check():
    print('Scope: nine-model matched validation (1050/309); CatBoost is controlled by experiments.py.',flush=True)
    protocol=json.loads((ROOT/'plans/protocol.json').read_text())
    for item in protocol['source_hashes']:
        if sha(item['path'])!=item['sha256']:raise RuntimeError(f"Source changed: {item['path']}")
    for split in protocol['folds']:
        if sha(split['image_checkpoint'])!=split['image_checkpoint_sha256']:raise RuntimeError('Image checkpoint changed')
    if (ROOT/'audit/frozen_source.json').exists():
        frozen=json.loads((ROOT/'audit/frozen_source.json').read_text())
        if sha(frozen['path'])!=frozen['sha256']:raise RuntimeError('Frozen source changed')
    if not RSCRIPT.exists():raise RuntimeError('R environment missing. See README.txt installation command.')
    subprocess.run([str(RSCRIPT),'-e','suppressPackageStartupMessages({library(glmnet);library(jsonlite);library(caret)}); cat("R dependencies OK\\n")'],
                   env=r_env(),check=True)
    print('Inputs, five saved mViT models and R environment verified.',flush=True)

def r_env():
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TZ='Asia/Shanghai')
    return env

def gpu_table():
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    result=[]
    for line in q.stdout.strip().splitlines():
        i,u,m,v=[x.strip() for x in line.split(',')]
        result.append({'index':i,'uuid':u,'memory_mib':int(m),'utilization':int(v)})
    return result

def resolve_gpus(value):
    wanted=[x.strip() for x in value.split(',') if x.strip()]
    table=gpu_table();out=[]
    for item in wanted:
        match=[x for x in table if item in [x['index'],x['uuid']]]
        if len(match)!=1:raise ValueError(f'Unknown GPU {item}')
        if match[0]['uuid'] not in [x['uuid'] for x in out]:out.append(match[0])
    if not out:raise ValueError('Provide one or more GPU indices, e.g. --gpus 0,3')
    return out

def receipt(path,stage,**info):
    write_json(path,{'protocol_sha256':sha(ROOT/'plans/protocol.json'),'stage':stage,'time':now(),**info})

def tabular(directory,final=False):
    directory=Path(directory)
    if (directory/'r_complete.json').exists():
        if not (directory/'r_receipt.json').exists():raise RuntimeError(f'Untracked R output in {directory}')
        previous=json.loads((directory/'r_receipt.json').read_text())
        if previous['protocol_sha256']!=sha(ROOT/'plans/protocol.json') or previous['code_sha256']!=sha(ROOT/'code/nested_tabular.R'):
            raise RuntimeError(f'R code or protocol changed since {directory.name}; inspect before reuse.')
        print('Reuse R stage:',directory.name,flush=True);return
    began=time.monotonic()
    command=[str(RSCRIPT),str(ROOT/'code/nested_tabular.R'),str(directory),str(ROOT/'plans/protocol.json')]
    if final:command.append('final')
    with (ROOT/'logs'/f'{directory.name}_R.log').open('a') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=r_env(),check=True)
    receipt(directory/'r_receipt.json','tabular',code_sha256=sha(ROOT/'code/nested_tabular.R'),seconds=time.monotonic()-began)
    print(f'R stage complete: {directory.name}, {time.monotonic()-began:.1f}s',flush=True)

def feature_is_complete(directory):
    if not (directory/'features_complete.json').exists():return False
    meta=json.loads((directory/'features_complete.json').read_text())
    spec=json.loads((directory/'split.json').read_text())
    if meta['checkpoint_sha256']!=spec['image_checkpoint_sha256']:raise RuntimeError('Stale image feature checkpoint')
    if meta.get('code_sha256')!=sha(ROOT/'code/extract_features.py'):raise RuntimeError('Image extraction code changed; inspect outputs')
    return (directory/'image_train.npz').exists() and (directory/'image_heldout.npz').exists()

def run_fusion_if_ready(fold):
    from fusion_stage import run_fusion
    directory=ROOT/'runs'/f'fold_{fold}'
    if (directory/'fusion_complete.json').exists():
        old=json.loads((directory/'fusion_receipt.json').read_text())
        if old['code_sha256']!=sha(ROOT/'code/fusion_stage.py') or old['protocol_sha256']!=sha(ROOT/'plans/protocol.json'):raise RuntimeError('Fusion code/protocol changed; inspect existing outputs')
        return True
    if not (directory/'r_complete.json').exists() or not feature_is_complete(directory):return False
    run_fusion(directory)
    receipt(directory/'fusion_receipt.json','fusion',code_sha256=sha(ROOT/'code/fusion_stage.py'))
    print(f'Fold {fold}: all nine fusion models complete.',flush=True);return True

def progress():
    rows=[]
    for fold in range(1,6):
        d=ROOT/'runs'/f'fold_{fold}'
        rows.append({'fold':fold,'tabular':(d/'r_complete.json').exists(),
                     'image_features':(d/'features_complete.json').exists(),'fusion':(d/'fusion_complete.json').exists()})
    result={'time':now(),'folds':rows,'complete':(ROOT/'FINAL_COMPLETE.json').exists()}
    write_json(ROOT/'progress.json',result);return result

def report():
    with (ROOT/'logs/report.log').open('a') as log:
        subprocess.run([PYTHON,str(ROOT/'code/report_nested.py')],stdout=log,stderr=subprocess.STDOUT,check=True)

def run(args):
    lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check()
    test=ROOT/'audit/software_validation.json'
    if not test.exists() or not json.loads(test.read_text()).get('passed'):raise RuntimeError('Run self-test first.')
    folds=[int(x) for x in args.folds.split(',')]
    if not set(folds)<=set(range(1,6)):raise ValueError('Folds must be 1 to 5')
    use_gpu=args.stage in ['all','features']
    use_cpu=args.stage in ['all','tabular']
    gpus=resolve_gpus(args.gpus) if use_gpu else []
    write_json(ROOT/'controller_process.json',{'pid':os.getpid(),'start':now(),'arguments':vars(args)})
    receipt(ROOT/'audit/last_execution.json','controller',arguments=vars(args),gpus=gpus,
            R_code_sha256=sha(ROOT/'code/nested_tabular.R'),feature_code_sha256=sha(ROOT/'code/extract_features.py'))
    status('running',stage_requested=args.stage,folds=folds,gpus=[g['index'] for g in gpus],cpu_workers=args.cpu_workers)
    tasks=queue.Queue()
    for fold in folds:
        if not feature_is_complete(ROOT/'runs'/f'fold_{fold}'):tasks.put(fold)
    futures=[]
    with ThreadPoolExecutor(max_workers=max(1,args.cpu_workers)) as cpu,ThreadPoolExecutor(max_workers=max(1,len(gpus))) as gpu:
        if use_cpu:
            futures += [cpu.submit(tabular,ROOT/'runs'/f'fold_{fold}') for fold in folds]
        if use_gpu:
            from gpu_dispatch import dispatch
            futures.append(gpu.submit(dispatch,gpus,folds,args))
        for completed in as_completed(futures):
            completed.result()
            if args.stage=='all':
                for fold in folds:run_fusion_if_ready(fold)
            progress()
    if args.stage in ['all','fusion']:
        for fold in folds:
            if not run_fusion_if_ready(fold):print(f'Fold {fold} awaits tabular/features; see progress.json.',flush=True)
    finished=all((ROOT/f'runs/fold_{i}/fusion_complete.json').exists() for i in range(1,6))
    if finished and args.stage in ['all','fusion']:
        from fusion_stage import prepare_final,run_fusion
        # Compute all development OOF thresholds before final external prediction.
        from report_nested import lock_thresholds
        lock_thresholds()
        directory=prepare_final()
        tabular(directory,final=True)
        if not (directory/'fusion_complete.json').exists():
            run_fusion(directory,final=True)
            receipt(directory/'fusion_receipt.json','final_fusion',code_sha256=sha(ROOT/'code/fusion_stage.py'))
        report()
        report_check=json.loads((ROOT/'audit/report_validation.json').read_text())
        if not report_check.get('completed'):raise RuntimeError('Final report did not validate')
        write_json(ROOT/'FINAL_COMPLETE.json',{'completed':True,'time':now(),
                   'protocol_sha256':sha(ROOT/'plans/protocol.json'),
                   'scope':'Nine-model matched fixed-fusion validation; CatBoost is a separate suite branch',
                   'report':'response/nested_validation_response_中文版.docx'})
        status('complete');progress()
    else:
        report();status('stage_complete',requested=args.stage,progress=progress())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check','self-test','run','launch','status','report'])
    parser.add_argument('--gpus',default='3',help='Physical GPU indices or UUIDs, comma separated; only inference uses GPU')
    parser.add_argument('--cpu-workers',type=int,default=4,choices=range(1,9))
    parser.add_argument('--folds',default='1,2,3,4,5')
    parser.add_argument('--stage',choices=['all','tabular','features','fusion'],default='all')
    parser.add_argument('--patient-batch',default='auto',help='auto benchmarks 1/2/4/8; or a positive integer')
    parser.add_argument('--slice-microbatch',type=int,default=32)
    parser.add_argument('--loader-workers',type=int,default=4)
    parser.add_argument('--allow-shared-gpu',action='store_true',default=True)
    parser.add_argument('--exclusive-gpu',dest='allow_shared_gpu',action='store_false')
    parser.add_argument('--gpu-workers-per-card',type=int,choices=range(1,5),default=2)
    parser.add_argument('--gpu-reserve-gib',type=float,default=4)
    parser.add_argument('--gpu-job-gib',type=float,default=8)
    args=parser.parse_args()
    if args.gpu_job_gib<=0 or args.gpu_reserve_gib<0:parser.error('GPU memory budgets must be positive (reserve may be zero)')
    if args.patient_batch!='auto' and int(args.patient_batch)<1:parser.error('patient batch must be positive')
    if args.slice_microbatch<1 or args.loader_workers<0:parser.error('Invalid microbatch/workers')
    if args.command=='prepare':
        from prepare import main as prepare
        prepare()
    elif args.command=='check':check()
    elif args.command=='self-test':
        from self_test import main as test
        test()
    elif args.command=='status':
        print(json.dumps(progress(),ensure_ascii=False,indent=2))
        if (ROOT/'status.json').exists():print((ROOT/'status.json').read_text())
    elif args.command=='report':report()
    elif args.command=='launch':
        # Verify that a running controller does not already hold the lock.
        with (ROOT/'controller.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'run',*sys.argv[2:]]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (ROOT/'logs/controller.log').open('a') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        write_json(ROOT/'launcher.json',{'pid':child.pid,'time':now(),'command':command})
        print(f'Controller launched, PID {child.pid}. Logs: {ROOT}/logs/controller.log')
    else:run(args)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc());raise
