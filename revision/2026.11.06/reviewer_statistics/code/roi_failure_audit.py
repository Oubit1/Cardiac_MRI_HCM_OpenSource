"""Recompute recorded reference-free fallback criteria on current masks; no label edits."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
import pandas as pd,numpy as np,nibabel as nib
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1]
def inspect(r):
 out={'cohort':r.cohort,'ID':r.ID,'slice_index':r.slice_index,'mask_path':r.mask_path,'recorded_reason':r.roi_reason}
 path=Path(r.mask_path)
 if not path.is_file():return {**out,'current_reason':'missing_mask'}
 mask=np.asarray(nib.load(path).dataobj)>0
 out['mask_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 if mask.shape!=(r.frame_count,r.height,r.width):return {**out,'current_reason':'mask_shape_mismatch'}
 area=mask.sum(axis=(1,2));frac=float(np.median(mask.mean(axis=(1,2))))
 comps=float(np.median([ndimage.label(m)[1] for m in mask]));den=area[:-1]+area[1:]
 dice=float(np.median((2*np.logical_and(mask[:-1],mask[1:]).sum(axis=(1,2))+1)/(den+1)))
 flags={'empty_frame':bool((area==0).any()),'area_fraction_outside_range':not .002<=frac<=.30,'median_components_gt2':comps>2,'median_temporal_dice_lt05':dice<.5}
 return {**out,'frames':len(mask),'empty_frames':int((area==0).sum()),'median_mask_fraction':frac,'median_components':comps,'median_temporal_dice':dice,**flags,'current_reason':'mask_reference_free_qc_failed' if any(flags.values()) else 'pass'}
def main():
 df=pd.read_csv(ROOT.parent/'audit/cache_slice_index.csv').query('roi_fallback==True');out=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(inspect,df.itertuples(index=False)):out.append(row)
 out=pd.DataFrame(out);out.to_csv(ROOT/'audit/ROI_fallback_criteria_recomputed.csv',index=False)
 rows=[]
 for cohort,g in out.groupby('cohort'):
  for flag in ['empty_frame','area_fraction_outside_range','median_components_gt2','median_temporal_dice_lt05']:
   rows.append({'cohort':cohort,'criterion':flag,'fallback_slices':len(g),'flagged_slices':int(g[flag].sum()),'fraction_of_fallback_slices':g[flag].mean()})
 pd.DataFrame(rows).to_csv(ROOT/'tables/ROI_fallback_criteria_summary.csv',index=False)
 summary={'slices_rechecked':len(out),'same_failure_classification':int((out.current_reason==out.recorded_reason).sum()),'current_masks_only':True,'labels_modified':False,'criterion_counts_overlap':True,'criterion_definition':'Original preprocessing: any empty phase; median foreground fraction outside [0.002,0.30]; median 4-connected components >2; median adjacent-frame smoothed Dice <0.5. These are reference-free triggers, not segmentation accuracy measures.'}
 (ROOT/'audit/ROI_fallback_criteria_audit.json').write_text(json.dumps(summary,indent=2));print(summary);print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
