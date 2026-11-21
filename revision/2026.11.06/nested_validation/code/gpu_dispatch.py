"""Memory-aware, bounded multi-process GPU dispatch; one lock per fold."""
from settings import *
import subprocess,time,fcntl

def devices():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
    result={}
    for row in p.stdout.splitlines():
        index,uuid,free,util=[x.strip() for x in row.split(',')]
        result[uuid]={'index':index,'uuid':uuid,'free_mib':int(free),'utilization':int(util)}
    return result

def eligible(device,active_count,args):
    # Reserve a full extra budget for already launched jobs while their allocation grows.
    return active_count<args.gpu_workers_per_card and device['free_mib'] >= 1024*(args.gpu_reserve_gib+args.gpu_job_gib*(active_count+1))

def dispatch(gpus,folds,args):
    from run_validation import feature_is_complete
    pending=list(folds);active=[];history=[]
    with (ROOT/'gpu_dispatch.lock').open('w') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while pending or active:
                for job in list(active):
                    rc=job['process'].poll()
                    if rc is None:continue
                    job['log'].close();job['lock'].close();active.remove(job)
                    history.append({'fold':job['fold'],'gpu':job['gpu'],'seconds':time.monotonic()-job['began'],'returncode':rc})
                    if rc:raise RuntimeError(f"GPU task fold {job['fold']} failed; see fold GPU log")
                    print(f"GPU {job['gpu']}: fold {job['fold']} complete",flush=True)
                pending=[f for f in pending if not feature_is_complete(ROOT/f'runs/fold_{f}')]
                table=devices()
                # Round-robin slots: spread one job per GPU before filling second slots.
                for slot in range(args.gpu_workers_per_card):
                    for gpu in gpus:
                        if not pending:break
                        d=table[gpu['uuid']];count=sum(x['uuid']==gpu['uuid'] for x in active)
                        if count!=slot or not eligible(d,count,args):continue
                        if not args.allow_shared_gpu and d['utilization']>10:continue
                        fold=pending[0];lock=(ROOT/f'runs/fold_{fold}/gpu_task.lock').open('w')
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:lock.close();continue
                        command=[PYTHON,str(ROOT/'code/extract_features.py'),'--fold',str(fold),'--gpu',gpu['uuid'],
                           '--patient-batch',args.patient_batch,'--slice-microbatch',str(args.slice_microbatch),'--loader-workers',str(args.loader_workers)]
                        log=(ROOT/f'logs/fold_{fold}_GPU.log').open('a')
                        try:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        except Exception:log.close();lock.close();raise
                        pending.pop(0);active.append({'process':process,'log':log,'lock':lock,'fold':fold,'gpu':d['index'],'uuid':gpu['uuid'],'began':time.monotonic()})
                        print(f"GPU {d['index']}: launch fold {fold}, slot {count+1}/{args.gpu_workers_per_card}, free {d['free_mib']} MiB",flush=True)
                write_json(ROOT/'gpu_dispatch_status.json',{'time':now(),'pending':pending,'active':[{'fold':j['fold'],'gpu':j['gpu'],'pid':j['process'].pid} for j in active],'finished':history})
                if pending or active:time.sleep(3)
        finally:
            # Do not abandon our children or release fold locks while they write outputs.
            for job in active:
                job['process'].wait();job['log'].close();job['lock'].close()
