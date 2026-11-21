"""Compare execution kernels on GPU 1; never train or select on validation outcomes."""
from common import ROOT, GPU_UUID, write_json, timestamp
from train import make_model, seed_all, set_encoder_trainable, Patients
import gc
import json
import time
import numpy as np
import torch


def configure(model, sdpa, checkpoint):
    model.use_gradient_checkpointing = checkpoint
    for module in model.modules():
        if module.__class__.__name__ == 'MultiScaleAttention':
            module.use_sdpa = sdpa


def main():
    seed_all(1729)
    model = make_model()
    model.train()
    set_encoder_trainable(model, True)
    records = json.loads((ROOT/'inputs/prepared_patients.json').read_text())
    index = next(i for i,r in enumerate(records) if r['cohort']=='development' and len(r['full_paths'])==8)
    videos, label, _ = Patients(records, [index])[0]
    videos = videos.cuda()
    target = torch.tensor(label, device='cuda')
    results = []
    baseline = None
    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logit, _ = model(videos, torch.empty(0, device='cuda'))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
        (loss*1024).backward()
        return logit, loss
    for sdpa, checkpoint in [(False,True),(False,False),(True,True),(True,False)]:
        configure(model,sdpa,checkpoint)
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        row = {'sdpa':sdpa,'gradient_checkpointing':checkpoint}
        try:
            seed_all(1729)
            logit,loss = step()
            torch.cuda.synchronize()
            gradients = torch.cat([p.grad.detach().float().cpu().flatten()/1024 for p in model.parameters() if p.grad is not None]).double()
            row.update(logit=float(logit.detach()),loss=float(loss.detach()),finite_gradients=bool(torch.isfinite(gradients).all()))
            if baseline is None:
                baseline = (row['logit'],gradients)
            row['logit_absolute_difference'] = abs(row['logit']-baseline[0])
            row['gradient_relative_l2_difference'] = float(torch.linalg.vector_norm(gradients-baseline[1])/torch.linalg.vector_norm(baseline[1]))
            row['gradient_cosine_similarity'] = float(torch.nn.functional.cosine_similarity(gradients,baseline[1],dim=0))
            del gradients,logit,loss
            timings=[]
            for _ in range(5):
                torch.cuda.synchronize(); began=time.monotonic()
                step()
                torch.cuda.synchronize(); timings.append(time.monotonic()-began)
            row.update(median_forward_backward_seconds=float(np.median(timings)),timings=timings,
                       max_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                       max_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            model.eval()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                before=time.monotonic()
                for _ in range(5): model(videos,torch.empty(0,device='cuda'))
                torch.cuda.synchronize()
            row['inference_seconds']=(time.monotonic()-before)/5
            model.train()
        except torch.cuda.OutOfMemoryError as exc:
            row['error']='CUDA out of memory'
            model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
        results.append(row)
        write_json(ROOT/'audit/acceleration_benchmark.json',{'time':timestamp(),'gpu_uuid':GPU_UUID,'patient_slices':8,
                   'purpose':'Execution comparison only; no optimizer update and no outcome-based selection','results':results})
        print(json.dumps(row),flush=True)


if __name__=='__main__': main()
