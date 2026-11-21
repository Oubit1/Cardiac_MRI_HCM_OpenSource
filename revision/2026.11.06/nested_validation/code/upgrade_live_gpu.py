"""Migrate only pending GPU work; keep all existing R and inference child processes alive."""
from settings import *
import psutil,signal,time
from types import SimpleNamespace
from gpu_dispatch import dispatch
from run_validation import resolve_gpus,feature_is_complete

def main():
    pid=json.loads((ROOT/'controller_process.json').read_text())['pid']
    p=psutil.Process(pid)
    assert str(ROOT/'code/run_validation.py') in p.cmdline(),p.cmdline()
    args=SimpleNamespace(gpu_workers_per_card=2,gpu_reserve_gib=4,gpu_job_gib=8,
                         allow_shared_gpu=True,patient_batch='auto',slice_microbatch=32,loader_workers=2)
    gpus=resolve_gpus('3,0')
    os.kill(pid,signal.SIGSTOP)
    try:
        running=[]
        for child in p.children(recursive=False):
            cmd=child.cmdline()
            if str(ROOT/'code/extract_features.py') in cmd:
                running.append((child,int(cmd[cmd.index('--fold')+1])))
        active_folds={f for _,f in running}
        pending=[f for f in range(1,6) if f not in active_folds and not feature_is_complete(ROOT/f'runs/fold_{f}')]
        write_json(ROOT/'audit/live_gpu_upgrade.json',{'time':now(),'controller_pid':pid,'existing_gpu_folds':sorted(active_folds),'new_gpu_folds':pending,'policy':vars(args)})
        print(f'Existing GPU folds {active_folds}; dispatch remaining {pending}; R children continue.',flush=True)
        dispatch(gpus,pending,args)
        # Old controller will reap its own GPU child and continue fusion/CPU scheduling.
    finally:
        if p.is_running():os.kill(pid,signal.SIGCONT)
        print('Original controller resumed; CPU and fusion flow retained.',flush=True)

if __name__=='__main__':main()
