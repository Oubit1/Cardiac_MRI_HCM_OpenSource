"""Single-GPU, resumable patient-level mViT training and locked inference."""
from common import ROOT, pin_gpu, status, write_json, sha256, timestamp
pin_gpu()
import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, str(ROOT / 'code/vendor'))
from legacy_trainer import FoundationAttentionMIL, set_encoder_trainable
from model_factory import Cardiac_MRI_Encoder


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def make_model(microbatch=8):
    encoder = Cardiac_MRI_Encoder('mvit', 50, 512, 1, None, 16, 'transfer', False)
    payload = torch.load(ROOT / 'inputs/pretrained.ckpt', map_location='cpu', weights_only=False)
    incoming = payload['state_dict']
    actual = encoder.state_dict()
    selected = {k.removeprefix('model.'): v for k, v in incoming.items() if k.removeprefix('model.') in actual}
    if set(selected) != set(actual):
        raise RuntimeError(f'Incomplete pretrained initialization: {len(selected)} of {len(actual)} tensors.')
    encoder.load_state_dict(selected, strict=True)
    del payload, incoming, selected
    model = FoundationAttentionMIL(encoder, 0, .25, microbatch).cuda()
    policy_path = ROOT/'plans/execution.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    model.use_gradient_checkpointing = policy.get('gradient_checkpointing', True)
    for module in model.modules():
        if module.__class__.__name__ == 'MultiScaleAttention':
            module.use_sdpa = policy.get('sdpa', False)
    return model


class Patients(Dataset):
    def __init__(self, records, indices, mode='full', augment=False, temporal='full'):
        self.records, self.indices = records, list(indices)
        self.mode, self.augment, self.temporal = mode, augment, temporal

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row = self.records[self.indices[item]]
        videos = []
        for path in row['roi_paths' if self.mode == 'roi' else 'full_paths']:
            cine = np.load(path, mmap_mode='r')
            count = len(cine)
            if self.temporal == 'first_frame':
                indices = np.zeros(16, dtype=int)
            else:
                indices = torch.linspace(0, count-1, 16).round().long().numpy()
                if self.augment and count > 16:
                    phase = int(torch.randint(0, count, ()).item())
                    indices = (indices-phase) % count
            frames = torch.from_numpy(np.array(cine[indices], copy=True))
            h, w = frames.shape[-2:]
            if self.augment:
                top, left = int(torch.randint(0, h-224+1, ()).item()), int(torch.randint(0, w-224+1, ()).item())
            else:
                top, left = (h-224)//2, (w-224)//2
            frames = frames[:, top:top+224, left:left+224]
            if self.augment and torch.rand(()) < .5:
                frames = frames.flip(-1)
            if self.augment:
                frames = (frames*(.9+.2*torch.rand(()))).clamp(0, 1)
            videos.append(frames[None].repeat(3, 1, 1, 1))
        return torch.stack(videos), float(row['event']), int(row['ID'])


def first(batch):
    return batch[0]


def loader(records, indices, config, augment, epoch=0):
    generator = torch.Generator().manual_seed(config['seed']*10000 + epoch*10 + int(augment))
    return DataLoader(Patients(records, indices, config['mode'], augment, config.get('temporal', 'full')),
                      batch_size=1, shuffle=augment, collate_fn=first, num_workers=2,
                      pin_memory=True, generator=generator, prefetch_factor=2)


@torch.inference_mode()
def predict(model, records, indices, config, epoch=0, features=False):
    model.eval()
    rows, vectors = [], []
    for videos, label, ident in loader(records, indices, config, False, epoch):
        videos = videos.cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logit, _, representation = model(videos, torch.empty(0, device='cuda'), return_features=True)
        rows.append({'ID': ident, 'event': int(label), 'probability': float(torch.sigmoid(logit).cpu())})
        if features:
            vectors.append(representation.float().cpu().numpy())
    frame = pd.DataFrame(rows)
    return frame, np.stack(vectors) if features else None


def eval_metrics(frame):
    return {'auc': float(roc_auc_score(frame.event, frame.probability)),
            'loss': float(log_loss(frame.event, np.clip(frame.probability, 1e-7, 1-1e-7), labels=[0, 1]))}


def save_torch(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp.pt')
    torch.save(payload, tmp)
    tmp.replace(path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(rng):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state(rng['cuda'])


def train_job(records, train_indices, validation_indices, config, directory):
    """Inner jobs select epochs only on inner validation; refits have no validation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = {**config, 'train_ids': [records[i]['ID'] for i in train_indices],
           'validation_ids': [records[i]['ID'] for i in validation_indices] if validation_indices is not None else [],
           'protocol_sha256': sha256(ROOT / 'plans/protocol.json')}
    cfg_path = directory / 'config.json'
    if cfg_path.exists():
        old_cfg = json.loads(cfg_path.read_text())
        if old_cfg != cfg:
            protocol = json.loads((ROOT/'plans/protocol.json').read_text())
            approved_hashes = protocol.get('resume_compatible_protocol_hashes', [])
            comparable = {**old_cfg, 'protocol_sha256': cfg['protocol_sha256']}
            if comparable != cfg or old_cfg['protocol_sha256'] not in approved_hashes:
                raise RuntimeError(f'Configuration changed for {directory}; choose a new run ID.')
            # Preserve the original scientific run configuration; execution changes are logged separately.
            cfg = old_cfg
    write_json(cfg_path, cfg)
    done = directory / 'complete.json'
    if done.exists():
        return json.loads(done.read_text())
    if validation_indices is None and not (directory/'latest.pt').exists():
        # Only reuse a fully completed, exactly identical refit (including patient order and seed).
        for source in sorted(directory.parent.glob('*/config.json')):
            if source.parent == directory or json.loads(source.read_text()) != cfg:
                continue
            if not (source.parent/'complete.json').exists() or not (source.parent/'final.pt').exists():
                continue
            result = json.loads((source.parent/'complete.json').read_text())
            os.link(source.parent/'final.pt', directory/'final.pt')
            shutil.copy2(source.parent/'history.csv',directory/'history.csv')
            result.update(reused_from=str(source.parent.relative_to(ROOT)),
                          source_training_seconds=result['total_seconds'], total_seconds=0)
            write_json(done,result)
            status('identical_refit_reused',run=directory.name,source=result['reused_from'])
            return result
    seed_all(config['seed'])
    model = make_model(config.get('microbatch', 8))
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder.parameters()), 'lr': config['encoder_lr']},
        {'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': 1e-4}],
        weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
    scaler = torch.amp.GradScaler('cuda')
    history, best = [], {'20': {'auc': -1, 'epoch': 0}, '60': {'auc': -1, 'epoch': 0}}
    latest = directory / 'latest.pt'
    start = 1
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        history, best, start = saved['history'], saved['best'], saved['epoch']+1
        restore_rng(saved['rng'])
        del saved
    execution_path = directory/'execution_sessions.json'
    sessions = json.loads(execution_path.read_text()) if execution_path.exists() else []
    policy_path = ROOT/'plans/execution.json'
    sessions.append({'started_at':timestamp(),'start_epoch':start,
                     'protocol_at_execution':sha256(ROOT/'plans/protocol.json'),
                     'execution':json.loads(policy_path.read_text()) if policy_path.exists() else {'sdpa':False,'gradient_checkpointing':True}})
    write_json(execution_path,sessions)
    for epoch in range(start, config['epochs']+1):
        began = time.monotonic()
        model.train()
        set_encoder_trainable(model, epoch > 2)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        n = len(train_indices)
        for step, (videos, label, _) in enumerate(loader(records, train_indices, config, True, epoch)):
            videos = videos.cuda(non_blocking=True)
            target = torch.tensor(label, device='cuda')
            group_size = min(4, n - (step//4)*4)
            with torch.autocast('cuda', dtype=torch.float16):
                logit, _ = model(videos, torch.empty(0, device='cuda'))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {directory.name}, epoch={epoch}, step={step}')
            scaler.scale(loss/group_size).backward()
            if (step+1) % 4 == 0 or step+1 == n:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if (step+1) % 100 == 0:
                status('training', run=directory.name, epoch=epoch, epoch_cap=config['epochs'],
                       patient_step=step+1, patients=n, gpu='GPU 1',
                       allocated_gb=round(torch.cuda.max_memory_allocated()/1024**3, 2))
        row = {'epoch': epoch, 'train_augmented_loss': float(np.mean(losses)),
               'encoder_lr': config['encoder_lr'], 'head_lr': 1e-4, 'encoder_trainable': epoch > 2}
        if validation_indices is not None:
            for name, indices in [('train', train_indices), ('validation', validation_indices)]:
                predictions, _ = predict(model, records, indices, config, epoch)
                row.update({name+'_'+k: v for k, v in eval_metrics(predictions).items()})
            for cap in [20, 60]:
                if epoch <= cap and row['validation_auc'] > best[str(cap)]['auc'] + 1e-4:
                    best[str(cap)] = {'auc': row['validation_auc'], 'epoch': epoch}
                    save_torch(directory / f'best_{cap}.pt',
                               {'model': model.state_dict(), 'epoch': epoch, 'config': cfg})
        row['seconds'] = time.monotonic()-began
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
        save_torch(latest, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                           'scaler': scaler.state_dict(), 'epoch': epoch, 'history': history,
                           'best': best, 'rng': rng_state()})
        status('epoch_complete', run=directory.name, **row)
        if validation_indices is not None and (epoch == 1 or epoch % 5 == 0):
            with (ROOT/'logs/report.log').open('a') as output:
                completed = subprocess.run([sys.executable,str(ROOT/'code/report.py')],cwd=ROOT,
                                           stdout=output,stderr=subprocess.STDOUT)
            if completed.returncode:
                print('Report refresh failed; training checkpoint is safe. See logs/report.log.',flush=True)
    if validation_indices is None:
        save_torch(directory / 'final.pt', {'model': model.state_dict(), 'epoch': config['epochs'], 'config': cfg})
    result = {'best': best, 'epochs': config['epochs'], 'total_seconds': sum(x['seconds'] for x in history),
              'validation_role': 'inner checkpoint selection' if validation_indices is not None else 'fixed-epoch refit'}
    write_json(done, result)
    del model, optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def infer_job(records, indices, config, weight_path, destination, features=False):
    destination = Path(destination)
    if destination.exists() and (not features or destination.with_suffix('.npz').exists()):
        return
    model = make_model(config.get('microbatch', 8))
    payload = torch.load(weight_path, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    frame, vectors = predict(model, records, indices, config, features=features)
    frame.to_csv(destination, index=False)
    if features:
        np.savez_compressed(destination.with_suffix('.npz'), ID=frame.ID.to_numpy(), features=vectors)
    del model, payload
    gc.collect()
    torch.cuda.empty_cache()


def smoke():
    """One optimizer update on synthetic images, not an experiment result."""
    seed_all(42)
    assert torch.cuda.device_count() == 1, 'More than one CUDA device visible.'
    model = make_model(8)
    model.train()
    set_encoder_trainable(model, True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    videos = torch.rand(8, 3, 16, 224, 224, device='cuda')
    started = time.monotonic()
    with torch.autocast('cuda', dtype=torch.float16):
        score, _ = model(videos, torch.empty(0, device='cuda'))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, torch.tensor(1., device='cuda'))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    write_json(ROOT / 'audit/gpu_smoke.json', {'device': torch.cuda.get_device_name(), 'visible_devices': 1,
               'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'seconds': time.monotonic()-started,
               'peak_allocated_gb': torch.cuda.max_memory_allocated()/1024**3,
               'synthetic_loss': float(loss.detach().cpu()), 'role': 'software smoke only'})
    print((ROOT / 'audit/gpu_smoke.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--pipeline-smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.pipeline_smoke:
        from validate_pipeline import validate
        validate()
