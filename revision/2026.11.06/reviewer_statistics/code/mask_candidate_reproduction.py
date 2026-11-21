import os
"""Small CPU inference to test candidate-output agreement, NOT reference accuracy."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd,nibabel as nib,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('JMRI_SEGMENTATION_ROOT', 'Automated_Annotation_of_Cine'))
sys.path.insert(0,str(SOURCE))
from cine_seg.data import load_image_hwt,normalize_frames,resize_frames
from cine_seg.model import UNet2D

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',default=str(ROOT/'audit/ROI_example_manifest.csv'));parser.add_argument('--tag',default='mask_candidate_reproduction');args=parser.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 samples=pd.read_csv(args.manifest);rows=[];sources=[]
 for run in ['unet2d_myo_all_centers','reproduce_cmr']:
  path=SOURCE/'runs'/run/'weights/best_model.pt';payload=torch.load(path,map_location='cpu',weights_only=False);config=payload['config']
  model=UNet2D(base_channels=int(config['model']['base_channels']));model.load_state_dict(payload['model_state'],strict=True);model.eval()
  sources.append({'run':run,'epoch':int(payload['epoch']),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  for r in samples.itertuples(index=False):
   image,_,axis=load_image_hwt(r.image_path);original=torch.from_numpy(np.moveaxis(image,-1,0).copy()).float()
   frames=resize_frames(normalize_frames(original),tuple(config['model']['image_size']))
   preds=[]
   with torch.inference_mode():
    for chunk in frames.split(4):
     prob=torch.sigmoid(model(chunk[:,None]))
     if prob.shape[-2:]!=original.shape[-2:]:prob=F.interpolate(prob,size=original.shape[-2:],mode='bilinear',align_corners=False)
     preds.append((prob[:,0]>=.5).numpy())
   pred=np.concatenate(preds);existing=np.moveaxis(np.asarray(nib.load(r.mask_path).dataobj)>0,axis,0);assert pred.shape==existing.shape
   inter=np.logical_and(pred,existing).sum();den=pred.sum()+existing.sum();disagree=np.logical_xor(pred,existing)
   rows.append({'run':run,'cohort':r.cohort,'ID':int(r.ID),'slice_index':int(r.slice_index),'frames':len(pred),'agreement_Dice_to_existing_mask':2*inter/den if den else 1.,'mismatched_voxels':int(disagree.sum()),'exact_match':not bool(disagree.any()),'exact_matching_frames':int((~disagree.any(axis=(1,2))).sum()),'existing_mask_sha256':r.mask_sha256})
   pd.DataFrame(rows).to_csv(ROOT/f'audit/{args.tag}.csv',index=False)
   print(run,r.cohort,r.ID,r.slice_index,'done',flush=True)
 out={'completed':True,'candidates':sources,'device':'cpu','precision':'float32','threshold':.5,'volumes':len(samples),'selection':'same fixed outcome-independent examples as ROI Figure R8','labels_modified':False,'is_reference_accuracy':False,'provenance_confirmed':False,'interpretation':'Agreement with existing labels only. Non-exact agreement cannot exclude GPU precision, historical preprocessing or postprocessing differences. Agreement alone does not replace inference provenance.'}
 out['manifest']=args.manifest;out['selection']='deterministic cohort/ID/slice selection documented in manifest; not based on outcomes or agreement'
 (ROOT/f'audit/{args.tag}.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
