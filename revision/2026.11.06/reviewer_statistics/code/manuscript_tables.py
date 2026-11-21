"""Publication tables matching supplied manuscript: TNR 12pt, black three-rule tables."""
from pathlib import Path
import json,numpy as np,pandas as pd
from docx import Document
from docx.shared import Pt,Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,Border,Side,Alignment,PatternFill
ROOT=Path(__file__).resolve().parents[1];CORE=ROOT.parent

def read(name):return pd.read_csv(ROOT/f'tables/{name}.csv')
def num(x):return 'NE' if pd.isna(x) else f'{float(x):.3f}'
def ci(row,key):
    if pd.isna(row.get(key,np.nan)):return 'NE'
    return f"{num(row[key])} [{num(row.get(key+'_CI_low',np.nan))}–{num(row.get(key+'_CI_high',np.nan))}]"
def pval(x):return 'NE' if pd.isna(x) else ('<0.001' if x<.001 else f'{x:.3f}')
def interval(row,value='estimate',lo='CI_low',hi='CI_high'):return f'{num(row[value])} [{num(row[lo])}–{num(row[hi])}]'
NAMES={'CMR':'Cardiac MRI (CatBoost)','Rscore_full':'Full Rscore','Rscore_texture':'Non-shape Rscore','CMR_Rscore_full':'Full Rscore + MRI (CatBoost)','CMR_Rscore_texture':'Non-shape Rscore + MRI (CatBoost)','CMR_Rscore_EN':'Full Rscore + MRI (elastic net)','Selected':'Inner-selected classifier strategy'}

class Tables:
 def __init__(self):
    self.doc=Document();sec=self.doc.sections[0];sec.page_width=Cm(29.7);sec.page_height=Cm(21);sec.left_margin=sec.right_margin=Cm(1.8);sec.top_margin=sec.bottom_margin=Cm(1.5)
    st=self.doc.styles['Normal'];st.font.name='Times New Roman';st.font.size=Pt(12);st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
    self.wb=Workbook();self.wb.remove(self.wb.active);self.index=[]
 def add(self,label,title,headers,rows,note,widths=None):
    if self.index:self.doc.add_page_break()
    table=self.doc.add_table(rows=2,cols=len(headers));table.autofit=False
    available=26.1
    widths=widths or ([6]+[(available-6)/(len(headers)-1)]*(len(headers)-1))
    for col,w in zip(table.columns,widths):col.width=Cm(w)
    titlecell=table.rows[0].cells[0].merge(table.rows[0].cells[-1]);titlecell.text=f'{label}. {title}'
    for cell,text in zip(table.rows[1].cells,headers):cell.text=text
    for row in rows:
     for cell,text in zip(table.add_row().cells,row):cell.text=str(text)
    foot=table.add_row().cells;foot[0].merge(foot[-1]).text=note
    n=len(table.rows)
    for i,row in enumerate(table.rows):
     if i in [0,1]:row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
     row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
     seen=set()
     for j,cell in enumerate(row.cells):
      if id(cell._tc) in seen:continue
      seen.add(id(cell._tc))
      borders=OxmlElement('w:tcBorders')
      for edge in ['top','left','bottom','right']:
       e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single' if ((i==1 and edge in ['top','bottom']) or (i==n-2 and edge=='bottom')) else 'nil');e.set(qn('w:sz'),'8');e.set(qn('w:color'),'000000');borders.append(e)
      cell._tc.get_or_add_tcPr().append(borders)
      for p in cell.paragraphs:
       p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or i in [0,n-1] else WD_ALIGN_PARAGRAPH.CENTER
       p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(2)
       for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12);run.bold=i<2;run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    ws=self.wb.create_sheet(label.replace('Table ','')[:31]);ws.sheet_view.showGridLines=False
    ws.append([f'{label}. {title}']);ws.merge_cells(start_row=1,start_column=1,end_row=1,end_column=len(headers));ws.append(headers)
    for row in rows:ws.append(row)
    last=ws.max_row;ws.append([note]);ws.merge_cells(start_row=last+1,start_column=1,end_row=last+1,end_column=len(headers))
    line=Side(style='thin',color='000000')
    for row in ws:
     for cell in row:
      cell.font=Font(name='Times New Roman',size=12,bold=cell.row<=2,color='000000');cell.fill=PatternFill(fill_type=None)
      cell.alignment=Alignment(horizontal='left' if cell.column==1 or cell.row in [1,last+1] else 'center',vertical='center',wrap_text=True)
      cell.border=Border(top=line if cell.row==2 else Side(),bottom=line if cell.row in [2,last] else Side())
    for j,w in enumerate(widths,1):ws.column_dimensions[__import__('openpyxl').utils.get_column_letter(j)].width=w*5
    ws.row_dimensions[1].height=45;ws.row_dimensions[2].height=45;ws.row_dimensions[last+1].height=100
    for i in range(3,last+1):ws.row_dimensions[i].height=42
    ws.freeze_panes='B3';ws.sheet_properties.pageSetUpPr.fitToPage=True;ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A4;ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0;ws.print_title_rows='1:2'
    self.index.append({'label':label,'title':title,'rows':len(rows),'note':note})
 def save(self):
    self.doc.save(ROOT/'tables/补充统计表_稿件样式.docx');self.wb.save(ROOT/'tables/补充统计表_稿件样式.xlsx')
    (ROOT/'audit/table_index.json').write_text(json.dumps(self.index,ensure_ascii=False,indent=2))

def main():
 t=Tables();m=read('performance');pairs=read('shape_comparisons');notes='Values in brackets are 95% confidence intervals from 2000 patient-stratified bootstrap samples. Development estimates are outer-fold predictions (n=1059); external estimates use the previously evaluated cohort (n=309). External thresholds are development-derived and unchanged within subgroups. Rscore-only models use original fitted probabilities, without fold-specific restandardization. NE, not estimable.'
 rows=[]
 for name in NAMES:
  d=m[(m.cohort=='development')&(m.model==name)].iloc[0];e=m[(m.cohort=='external_test')&(m.model==name)].iloc[0]
  rows.append([NAMES[name],ci(d,'AUC'),ci(e,'AUC'),num(e.Sensitivity),num(e.Specificity),num(e.NPV),num(e.Brier)])
 t.add('Table R1','Performance of full and non-shape radiomics models',['Model','Development AUC','External AUC','Sensitivity','Specificity','NPV','Brier score'],rows,notes)
 rows=[[r.cohort,NAMES[r.A],NAMES[r.B],interval(r,'Delta_AUC_B_minus_A','Delta_CI_low','Delta_CI_high'),pval(r.Holm_P)] for _,r in pairs.iterrows()]
 t.add('Table R2','Paired comparisons for shape-feature ablation',['Cohort','Model A','Model B','ΔAUC (B − A) [95% CI]','Holm P'],rows,'Paired patients and bootstrap resamples; four comparisons corrected by Holm separately within each cohort. Removing shape features preceded all supervised selection.')
 op=read('operating_changes')
 t.add('Table R3','Changes in external classification at development-derived operating points',['Added score','Metric','Difference [95% CI]','Improved / worsened, n','Exact McNemar P'],[[NAMES[r.B],r.metric,interval(r,'delta'),f'{r.improved_n} / {r.worsened_n}',pval(r.McNemar_exact_P)] for _,r in op.iterrows()], 'The baseline is cardiac MRI-only CatBoost. Each model uses its own development-derived threshold; this is not an external sensitivity-matched specificity comparison. Improvement/worsening counts refer to correctness within the corresponding outcome group.')
 nri=read('NRI_IDI');rows=[]
 for (cohort,b),sub in nri.groupby(['cohort','B'],sort=False):
  z=sub.set_index('metric');rows.append([cohort,NAMES[b],*[interval(z.loc[k]) for k in ['NRI_event','NRI_nonevent','NRI_total','IDI']]])
 t.add('Table R4','Reclassification after adding the radiomics score',['Cohort','Model vs MRI-only','Event NRI','Non-event NRI','Total NRI','IDI'],rows,'Continuous NRI: event component = proportion moving up minus down; non-event component = proportion moving down minus up. Ties contribute zero. Positive values favor the added-score model. These are probability-change measures, not proportions of patients benefiting clinically. Brackets denote patient-stratified bootstrap 95% CIs.')
 pr=read('prevalence_scenarios');sub=pr[(pr.model=='CMR_Rscore_full')&pr.prevalence.isin([.3,.4,.5,.6,.7])]
 t.add('Table R5','Predictive values under hypothetical LGE prevalence',['LGE prevalence','NPV [95% CI]','PPV [95% CI]'],[[f'{r.prevalence:.0%}',ci(r,'NPV'),ci(r,'PPV')] for _,r in sub.iterrows()], 'Full-score CatBoost branch. Sensitivity and specificity are held at external estimates; intervals propagate their sampling variability with the threshold fixed. These are transportability scenarios, not validation in new populations.')
 sg=read('subgroup_performance')
 for label,factor in [('R6','field_strength_t'),('R7','resolution_group')]:
  sub=sg[(sg.factor==factor)&sg.model.isin(['CMR','Rscore_full','CMR_Rscore_full'])]
  t.add('Table '+label,'External performance by '+('magnetic field strength' if factor=='field_strength_t' else 'in-plane pixel spacing'),['Group','Model','n / LGE+','AUC [95% CI]','Sensitivity','Specificity','NPV'],[[r.group,NAMES[r.model],f'{int(r.N)} / {int(r.Events)}',ci(r,'AUC'),num(r.get('Sensitivity')),num(r.get('Specificity')),num(r.get('NPV'))] for _,r in sub.iterrows()], 'Thresholds are unchanged from full-development selection. Metadata cover all 309 external patients. Pixel spacing is the geometric mean of row and column spacing; the split is the development-cohort median, not selected by external performance. Center membership is unavailable and is not inferred from field strength.')
 sub=sg[(sg.factor.isin(['field_strength_t','resolution_group']))&(sg.model=='CMR_Rscore_full')]
 t.add('Table R8','External subgroup calibration of full-score CatBoost',['Grouping','Group','n','Intercept [95% CI]','Slope [95% CI]','Brier score'],[[r.factor,r.group,int(r.N),ci(r,'Calibration_intercept_joint'),ci(r,'Calibration_slope_joint'),num(r.Brier)] for _,r in sub.iterrows()], 'Calibration intercept and slope are jointly estimated. Intervals describe variability conditional on the fitted predictions and do not refit the full training pipeline.')
 reg=read('Rscore_adjusted_association');sub=reg[(reg.analysis=='field_pixel')&reg.term.isin(['event','pixel_mm','field_3T'])]
 t.add('Table R9','Exploratory associations between acquisition factors and Rscore',['Cohort','Variable','n','Coefficient [95% CI]','P value'],[[r.cohort,r.term,int(r.N),interval(r,'coefficient'),pval(r.P)] for _,r in sub.iterrows()], 'Outcome is the full Rscore scaled using the development OOF mean and SD. OLS with HC3 standard errors; development analyses adjust for fold, LGE and spacing; external analyses adjust for LGE, spacing and field strength. No center adjustment is possible. Vendor-adjusted exploratory results are retained in the detailed workbook pending source verification. Association is not causal evidence.')
 core=pd.read_excel(CORE/'tables/mViT_revision_results.xlsx',sheet_name='Performance');sub=core[(core.Dataset=='External test')&core.Model.isin(['Submitted mViT-FT','baseline20_full','extended60_full','roi_matched_full'])]
 t.add('Table R10','External mViT performance before and after ROI restriction',['Model','AUC [95% CI]','Sensitivity','Specificity','NPV','Brier score'],[[r.Model,ci(r,'AUC'),num(r.Sensitivity),num(r.Specificity),num(r.NPV),num(r.Brier)] for _,r in sub.iterrows()], 'Submitted mViT is a historical reference. The paired ROI comparison uses extended60_full versus roi_matched_full with matched training duration. ROI is mask-guided cropping, not strict myocardium-only masking. Available model-selection segmentation validation is reported separately in Table R11. NPV is not estimable if no patients are classified negative.')
 seg=read('segmentation_validation')
 t.add('Table R11','Patient-held-out segmentation validation in three private datasets',['Run / epoch','Center','Patients / volumes','Pooled Dice','IoU','Patient mean Dice [95% CI]'],[[f'{r.run} / {r.best_epoch}',r.center,f'{r.patients} / {r.volumes}',num(r.pooled_pixel_Dice),num(r.IoU),interval(r,'patient_mean_volume_Dice','patient_Dice_CI_low','patient_Dice_CI_high')] for _,r in seg.iterrows()], 'Both runs use 952 training and 106 validation patients. Validation Dice selected the checkpoint; these are model-selection validation results. Pooled Dice weights pixels; patient mean Dice averages volume Dice within each patient, then weights patients equally (2000 patient bootstrap samples). The checkpoint used to generate study ROI masks remains to be linked to these runs.')
 t.save()
 # Preserve complete numerical precision in the audit workbook, while matching manuscript typography.
 wb=load_workbook(ROOT/'tables/统计明细.xlsx')
 if 'Segmentation_validation' in wb.sheetnames:del wb['Segmentation_validation']
 ws=wb.create_sheet('Segmentation_validation');ws.append(list(seg.columns))
 for row in seg.itertuples(index=False,name=None):ws.append(list(row))
 for ws in wb:
  ws.sheet_view.showGridLines=False;ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions;last_row=ws.max_row
  for row in ws:
   for c in row:
    c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.fill=PatternFill(fill_type=None)
    c.alignment=Alignment(vertical='top',wrap_text=True)
    c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last_row] else Side())
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=22
 wb.save(ROOT/'tables/统计明细.xlsx')
 print('Manuscript-style Word and Excel tables written.')

if __name__=='__main__':main()
