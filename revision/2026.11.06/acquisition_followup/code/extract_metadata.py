from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np,pandas as pd,pydicom
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
TAGS=['SeriesInstanceUID','PixelBandwidth','PixelSpacing','Rows','Columns','SliceThickness','SpacingBetweenSlices','MagneticFieldStrength','RepetitionTime','EchoTime','FlipAngle','ImageType','Modality','Manufacturer','InstitutionName']
def number(v):
 try:return float(v)
 except (ValueError,TypeError):return np.nan
def read(r):
 d={'cohort':r.cohort,'ID':r.ID,'representative_dicom':r.representative_dicom}
 try:
  ds=pydicom.dcmread(r.representative_dicom,stop_before_pixels=True,specific_tags=TAGS);d['series_matches']=str(getattr(ds,'SeriesInstanceUID',''))==str(r.series_uid);d['read_status']='ok'
  d['image_type']='\\'.join(str(x) for x in getattr(ds,'ImageType',[]));d['modality']=str(getattr(ds,'Modality',''));d['header_manufacturer']=str(getattr(ds,'Manufacturer',''));d['header_institution']=str(getattr(ds,'InstitutionName',''))
  for field,tag in [('receiver_bandwidth_hz_px','PixelBandwidth'),('header_rows','Rows'),('header_columns','Columns'),('header_thickness_mm','SliceThickness'),('header_spacing_mm','SpacingBetweenSlices')]:d[field]=number(getattr(ds,tag,None))
  pix=getattr(ds,'PixelSpacing',[])
  d['reconstructed_fov_row_mm']=d['header_rows']*number(pix[0]) if len(pix)==2 else np.nan;d['reconstructed_fov_col_mm']=d['header_columns']*number(pix[1]) if len(pix)==2 else np.nan;d['reconstructed_fov_geomean_mm']=np.sqrt(d['reconstructed_fov_row_mm']*d['reconstructed_fov_col_mm'])
  if not d['series_matches']:
   for key in ['receiver_bandwidth_hz_px','reconstructed_fov_row_mm','reconstructed_fov_col_mm','reconstructed_fov_geomean_mm']:d[key]=np.nan
 except Exception as e:d['read_status']=type(e).__name__;d['series_matches']=False
 return d

def main():
 meta=pd.read_csv(CORE/'inputs/acquisition_metadata.csv');out=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for i,d in enumerate(pool.map(read,meta.itertuples(index=False)),1):
   out.append(d)
   if i%300==0:print('headers',i,flush=True)
 out=pd.DataFrame(out);out.to_csv(ROOT/'inputs/additional_cine_headers.csv',index=False)
 joined=meta.merge(out.drop(columns='representative_dicom'),on=['cohort','ID'],validate='one_to_one');joined['pixel_mm']=np.sqrt(joined.pixel_spacing_row_mm*joined.pixel_spacing_col_mm);joined['derived_gap_mm']=joined.spacing_between_slices_mm-joined.slice_thickness_mm
 joined.to_csv(ROOT/'inputs/extended_acquisition_metadata.csv',index=False)
 summary={'patients':len(joined),'readable_headers':int((out.read_status=='ok').sum()),'series_matched':int(out.series_matches.sum()),'bandwidth_nonmissing':int(joined.receiver_bandwidth_hz_px.notna().sum()),'derived_gap_negative_n':int((joined.derived_gap_mm<0).sum())};(ROOT/'audit/header_extraction.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
