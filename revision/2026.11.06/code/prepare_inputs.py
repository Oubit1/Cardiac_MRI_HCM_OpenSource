"""Map archived SAX slices to migrated cine assets, then cache deterministic preprocessing.

Only cine images and existing cine masks are read. LGE assets are never loaded.
Masks have incompletely documented origin: ROI experiments are explicitly exploratory.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from scipy import ndimage
from common import ROOT, DATASET, status, write_json, sha256


def resize(array):
    # Legacy normalize_cine casts to float32 BEFORE interpolation.
    array = array.astype(np.float32, copy=False)
    height, width = array.shape[-2:]
    scale = 244 / min(height, width)
    return F.interpolate(torch.from_numpy(array)[:, None],
                         size=(max(224, round(height*scale)), max(224, round(width*scale))),
                         mode='bilinear', align_corners=False)[:, 0].numpy()


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npy')
    np.save(temp, array.astype(np.float32, copy=False))
    temp.replace(path)


def prepare_patient(records):
    torch.set_num_threads(1)
    output = []
    for r in records:
        key = f"{r['cohort']}_{int(r['ID']):04d}_slice{int(r['slice_index']):03d}"
        dest = ROOT / 'cache/cine' / (key + '.npy')
        roi_dest = ROOT / 'cache/roi' / (key + '.npy')
        metadata = ROOT / 'cache/metadata' / (key + '.json')
        if metadata.exists():
            saved = json.loads(metadata.read_text())
            if saved.get('preprocessing_version') == 'float32_before_interpolation_v2' and dest.exists() and Path(saved['roi_cache']).exists():
                output.append(saved)
                continue
        cine = np.asarray(nib.load(r['current_path']).dataobj, dtype=np.float32)
        expected = (int(r['frame_count']), int(r['height']), int(r['width']))
        if cine.shape != expected:
            raise ValueError(f'{key}: migrated image shape {cine.shape} != archived {expected}')
        finite = cine[np.isfinite(cine)]
        if not len(finite):
            raise ValueError(f'{key}: no finite image intensities')
        lo, hi = np.percentile(finite, [1, 99])
        cine = np.nan_to_num(cine, nan=lo, posinf=hi, neginf=lo)
        cine = (np.clip(cine, lo, hi)-lo)/max(float(hi-lo), 1e-6)
        # Spatial interpolation and frame selection commute; save all native phases.
        full = resize(cine)
        save_array(dest, full)
        reason, bbox = '', None
        mask_path = r.get('mask_path', '')
        if not mask_path or not Path(mask_path).is_file():
            reason = 'missing_mask'
        else:
            mask = np.asarray(nib.load(mask_path).dataobj) > 0
            if mask.shape != cine.shape:
                reason = 'mask_shape_mismatch'
            else:
                area = mask.sum(axis=(1, 2))
                fraction = float(np.median(mask.mean(axis=(1, 2))))
                components = np.median([ndimage.label(m)[1] for m in mask])
                denom = area[:-1] + area[1:]
                dice = np.median((2*np.logical_and(mask[:-1], mask[1:]).sum(axis=(1, 2))+1)/(denom+1))
                if (area == 0).any() or not .002 <= fraction <= .30 or components > 2 or dice < .5:
                    reason = 'mask_reference_free_qc_failed'
                else:
                    yy, xx = np.where(mask.any(axis=0))
                    h, w = cine.shape[-2:]
                    cy, cx = (yy.min()+yy.max()+1)/2, (xx.min()+xx.max()+1)/2
                    side = int(np.ceil(max(yy.max()-yy.min()+1, xx.max()-xx.min()+1, 32)*1.4))
                    # 20% margin on each side of the time-union box; fixed over time.
                    side = min(side, h, w)
                    top = int(np.clip(round(cy-side/2), 0, h-side))
                    left = int(np.clip(round(cx-side/2), 0, w-side))
                    bbox = [top, left, side, side]
                    save_array(roi_dest, resize(cine[:, top:top+side, left:left+side].copy()))
        if reason:
            # Preserve the common patient set; no outcome-driven exclusions.
            roi_dest = dest
        record = {**r, 'full_cache': str(dest), 'roi_cache': str(roi_dest),
                  'preprocessing_version': 'float32_before_interpolation_v2',
                  'roi_fallback': bool(reason), 'roi_reason': reason, 'roi_bbox': bbox,
                  'cache_shape': list(full.shape), 'image_sha256': sha256(r['current_path']),
                  'normalization_low': float(lo), 'normalization_high': float(hi),
                  'mask_origin': 'existing_cine_mask_origin_not_fully_verified',
                  'review_use': 'cine-only; LGE association and registration are not used'}
        write_json(metadata, record)
        output.append(record)
    return output


def map_inputs():
    old = pd.read_csv(ROOT / 'inputs/historical_slice_qc.csv')
    rows = []
    for line in (DATASET / 'metadata/samples.jsonl').open():
        sample = json.loads(line)
        if sample['task'] != 'slice_cmr_lge':
            continue
        cohort, ident, layer = sample['sample_id'].rsplit('_', 2)
        asset = sample['images']['cmr']
        label = sample['labels'].get('cmr_myo', {})
        rows.append({'cohort': cohort, 'ID': int(ident), 'slice_index': int(layer.replace('slice', '')),
                     'current_path': str(DATASET / asset['path']),
                     'mask_path': str(DATASET / label['path']) if label else '',
                     'current_shape': asset['shape']})
    mapped = old.merge(pd.DataFrame(rows), on=['cohort', 'ID', 'slice_index'], how='outer',
                       indicator=True, validate='one_to_one')
    if not (mapped['_merge'] == 'both').all():
        raise ValueError('Migrated slices do not match the historical index.')
    mapped = mapped.drop(columns='_merge')
    selected = []
    for _, group in mapped.groupby(['cohort', 'ID'], sort=False):
        group = group.sort_values(['slice_coordinate', 'slice_index'])
        index = np.unique(np.linspace(0, len(group)-1, min(8, len(group))).round().astype(int))
        selected.append(group.iloc[index])
    selected = pd.concat(selected, ignore_index=True)
    selected.to_csv(ROOT / 'audit/selected_slice_mapping.csv', index=False)
    current = pd.read_csv(ROOT / 'inputs/analysis_manifest.csv')
    reference = pd.read_csv(ROOT / 'inputs/submitted_ft_development_oof_predictions.csv')
    ext_ref = pd.read_csv(ROOT / 'inputs/submitted_ft_external_test_predictions.csv')
    for cohort, ref in [('development', reference), ('external_test', ext_ref)]:
        joined = current[current.cohort == cohort].merge(ref[['ID', 'event']], on='ID', suffixes=('', '_ref'))
        assert (joined.event == joined.event_ref).all()
        assert set(selected[selected.cohort == cohort].ID) == set(ref.ID)
    return selected, current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit-patients', type=int, default=0)
    args = parser.parse_args()
    selected, current = map_inputs()
    groups = [g.to_dict('records') for _, g in selected.groupby(['cohort', 'ID'], sort=False)]
    if args.limit_patients:
        groups = groups[:args.limit_patients]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare_patient, group) for group in groups]
        for done, job in enumerate(as_completed(jobs), 1):
            rows.extend(job.result())
            if done % 25 == 0 or done == len(jobs):
                status('preparing_inputs', patients_done=done, patients_total=len(jobs))
    table = pd.DataFrame(rows).sort_values(['cohort', 'ID', 'slice_coordinate', 'slice_index'])
    table.to_csv(ROOT / 'audit/cache_slice_index.csv', index=False)
    patients = []
    for (cohort, ident), group in table.groupby(['cohort', 'ID'], sort=False):
        row = current[(current.cohort == cohort) & (current.ID == ident)].iloc[0].to_dict()
        row.update(full_paths=group.full_cache.tolist(), roi_paths=group.roi_cache.tolist(),
                   roi_fallback_slices=int(group.roi_fallback.sum()), slice_indices=group.slice_index.tolist())
        patients.append(row)
    write_json(ROOT / 'inputs/prepared_patients.json', patients)
    write_json(ROOT / 'audit/cache_summary.json', {'patients': len(patients), 'slices': len(table),
               'preprocessing_version': 'float32_before_interpolation_v2',
               'roi_fallback_slices': int(table.roi_fallback.sum()), 'limited_preflight': bool(args.limit_patients),
               'labels': 'Current analysis_manifest.csv, confirmed against submitted Table 4 predictions',
               'historical_development_folds': 'Reconstructed explicitly for revision replication; migrated train/val split not used',
               'roi_role': 'Exploratory localization with incompletely documented existing masks; not expert manual ROIs'})
    status('inputs_ready', patients=len(patients), slices=len(table))


if __name__ == '__main__':
    main()
