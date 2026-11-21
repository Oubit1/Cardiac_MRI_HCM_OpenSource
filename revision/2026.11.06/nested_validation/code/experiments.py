"""One controller for matched nine models, full-cohort model selection and shape ablation."""
from settings import *
import argparse,subprocess,sys,fcntl
from concurrent.futures import ThreadPoolExecutor
import tabular_suite as tab

def prepare():
    from prepare_frozen import main
    main();tab.prepare()

def report():
    for script in ['report_nested.py','report_tabular.py']:
        subprocess.run([PYTHON,str(ROOT/'code'/script)],check=True)

def execute(args):
    with (ROOT/'suite.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        tasks=['matched','catboost','no-shape'] if args.task=='all' else [args.task]
        write_json(ROOT/'suite_status.json',{'state':'running','pid':os.getpid(),'time':now(),'arguments':vars(args)})
        workers=args.cpu_workers
        def matched():
            cmd=[PYTHON,str(ROOT/'code/run_validation.py'),'run','--gpus',args.gpus,'--cpu-workers',str(max(1,workers//2) if len(tasks)>1 else workers),
                 '--patient-batch',args.patient_batch,'--slice-microbatch',str(args.slice_microbatch),'--folds',args.folds]
            cmd += ['--gpu-workers-per-card',str(args.gpu_workers_per_card),'--gpu-reserve-gib',str(args.gpu_reserve_gib),'--gpu-job-gib',str(args.gpu_job_gib)]
            if not args.allow_shared_gpu:cmd.append('--exclusive-gpu')
            with (ROOT/'logs/matched_controller.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        def cpu_tasks():
            count=max(1,workers-workers//2) if 'matched' in tasks else workers
            for task,variant in [('catboost','all_features'),('no-shape','no_shape')]:
                if task in tasks:tab.run(variant,count,[int(x) for x in args.folds.split(',')])
        if workers==1:
            if 'matched' in tasks:matched()
            cpu_tasks()
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs=[pool.submit(cpu_tasks)]
                if 'matched' in tasks:jobs.append(pool.submit(matched))
                for job in jobs:job.result()
        report()
        write_json(ROOT/'suite_status.json',{'state':'requested_stages_finished','time':now(),'arguments':vars(args),
                  'matched_complete':(ROOT/'FINAL_COMPLETE.json').exists(),
                  'tabular_complete':{v:(tab.BASE/v/'COMPLETE.json').exists() for v in ['all_features','no_shape']}})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','check','run','launch','status','report','test'])
    p.add_argument('--task',choices=['all','matched','catboost','no-shape'],default='all')
    p.add_argument('--gpus',default='0,3');p.add_argument('--cpu-workers',type=int,default=4,choices=range(1,9))
    p.add_argument('--folds',default='1,2,3,4,5');p.add_argument('--patient-batch',default='auto')
    p.add_argument('--slice-microbatch',type=int,default=32);p.add_argument('--allow-shared-gpu',action='store_true',default=True)
    p.add_argument('--exclusive-gpu',dest='allow_shared_gpu',action='store_false')
    p.add_argument('--gpu-workers-per-card',type=int,choices=range(1,5),default=2)
    p.add_argument('--gpu-reserve-gib',type=float,default=4)
    p.add_argument('--gpu-job-gib',type=float,default=8)
    a=p.parse_args()
    if a.gpu_job_gib<=0 or a.gpu_reserve_gib<0:p.error('Invalid GPU memory budget')
    if not set(a.folds.split(','))<=set('12345') or len(set(a.folds.split(',')))!=len(a.folds.split(',')):p.error('folds must be unique values 1 to 5')
    if a.command=='prepare':prepare()
    elif a.command=='check':
        from run_validation import check
        check();prepare()
        for family in tab.FAMILIES:tab.estimator(family,tab.candidates(family)[0])
        print('All nine-model, nested classifier and no-shape dependencies ready.')
    elif a.command=='status':
        for file in ['suite_status.json','progress.json']:
            path=ROOT/file
            if path.exists():print(path.read_text())
        for variant in ['all_features','no_shape']:
            print(variant,{i:(tab.BASE/variant/f'fold_{i}/complete.json').exists() for i in range(1,6)})
            print('  completed R stages:',len(list((tab.BASE/variant).rglob('upstream_receipt.json'))),'/ 31')
    elif a.command=='report':report()
    elif a.command=='test':
        for script in ['self_test.py','test_report.py','test_tabular_suite.py']:subprocess.run([PYTHON,str(ROOT/'code'/script)],check=True)
    elif a.command=='launch':
        with (ROOT/'suite.lock').open('w') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'run',*sys.argv[2:]]
        with (ROOT/'logs/experiments.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(ROOT/'suite_launcher.json',{'pid':child.pid,'command':cmd,'time':now()})
        print(f'PID {child.pid}; log: {ROOT}/logs/experiments.log')
    else:execute(a)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(ROOT/'suite_error.json',{'time':now(),'error':repr(exc)});raise
