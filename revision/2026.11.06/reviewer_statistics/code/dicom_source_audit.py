"""Bounded read-only external DICOM metadata audit; never infer B/C from scanner."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,json,hashlib
import pandas as pd,numpy as np,pydicom
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent
TAGS=['InstitutionName','InstitutionalDepartmentName','Manufacturer','ManufacturerModelName','MagneticFieldStrength','StationName','DeviceSerialNumber','SeriesInstanceUID']
def scan(row):
 files=[]
 for base,dirs,names in os.walk(row.source_path):
  dirs.sort()
  files.extend(str(Path(base)/n) for n in sorted(names) if not n.startswith('.') and n.lower().endswith('.dcm'))
 if not files:
  for base,dirs,names in os.walk(row.source_path):
   dirs.sort();files.extend(str(Path(base)/n) for n in sorted(names) if not n.startswith('.'))
 chosen=sorted(set([str(row.representative_dicom)]+[files[i] for i in np.linspace(0,len(files)-1,min(16,len(files))).astype(int)])) if files else [str(row.representative_dicom)]
 out=[]
 for f in chosen:
  rec={'ID':int(row.ID),'path':f,'sampling':'representative plus up to 16 evenly spaced sorted files'}
  try:
   ds=pydicom.dcmread(f,stop_before_pixels=True,specific_tags=TAGS)
   for k in TAGS:
    v=str(getattr(ds,k,''))
    # Device and series identifiers are useful for consistency but need not be displayed verbatim.
    rec[k]=hashlib.sha256(v.encode()).hexdigest()[:16] if k in ['DeviceSerialNumber','SeriesInstanceUID'] and v else v
   rec['status']='ok'
  except Exception as e:rec['status']=type(e).__name__
  out.append(rec)
 return out

def main():
 data=pd.read_csv(CORE/'inputs/acquisition_metadata.csv').query("cohort=='external_test'");out=[]
 with ThreadPoolExecutor(max_workers=8) as pool:
  for rows in pool.map(scan,data.itertuples(index=False)):out.extend(rows)
 df=pd.DataFrame(out);df.to_csv(ROOT/'audit/external_DICOM_metadata_samples.csv',index=False)
 def meaningful(s):return sorted({str(v).strip() for v in s.dropna() if str(v).strip().lower() not in ['', 'ano','anonymous','anonymized','anon','none','nan','unknown']})
 rows=[]
 for pid,g in df.groupby('ID'):
  ok=g[g.status=='ok'];r={'ID':pid,'sampled_headers':len(g),'readable_headers':len(ok)}
  for key in TAGS:r[key+'_values']=' | '.join(meaningful(ok[key]))
  r['candidate_institution_found']=bool(r['InstitutionName_values']);rows.append(r)
 patients=pd.DataFrame(rows);patients.to_csv(ROOT/'tables/external_center_metadata_audit.csv',index=False)
 summary={'patients':len(patients),'headers_sampled':len(df),'headers_readable':int((df.status=='ok').sum()),'patients_with_candidate_institution':int(patients.candidate_institution_found.sum()),'sampling':'Existing SAX representative plus up to 16 evenly spaced sorted DICOM filenames per patient. Not an exhaustive all-header search.','B_C_mapping_established':False,'manufacturers':sorted(set(v for s in patients.Manufacturer_values for v in s.split(' | ') if v))}
 (ROOT/'audit/external_center_metadata_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
