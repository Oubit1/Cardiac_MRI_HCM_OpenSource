from pathlib import Path
import json,sys
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font,Alignment,Border,Side
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1];R=ROOT.parent/'nested_validation/tabular_full/all_features'
sys.path.insert(0,str(ROOT/'code'))
from manuscript_tables import Tables
oldpath=PROJECT/'Data/Radiomics/Rscore_OOF_results/Rscore_nested_OOF_results.xlsx'
old=pd.read_excel(oldpath,sheet_name='Final_coefficients_standardized');oldset=set(old.Feature)-{'(Intercept)'}
summary=[];details=[];sets=[]
for fold in [1,2,3,4,5,'Final refit']:
 d=R/'final' if isinstance(fold,str) else R/f'fold_{fold}/outer'
 c=json.loads((d/'cmr_selection/selection.json').read_text());r=json.loads((d/'rscore_full/selection.json').read_text());rad=set(v.removeprefix('rad__') for v in r['selected_features']);sets.append(rad)
 summary.append({'stage':fold,'CMR_n':len(c['selected_features']),'CMR_features':', '.join(v.removeprefix('cmr__') for v in c['selected_features']),'Rscore_n':len(rad),'shape_n':sum('_shape_' in v for v in rad),'overlap_with_archived_23':len(rad&oldset),'Rscore_alpha':r['alpha'],'Rscore_lambda':r['lambda'],'CMR_lambda':c['lambda']})
 for kind in ['cmr_selection','rscore_full']:
  df=pd.read_csv(d/kind/'coefficients.csv');df['stage']=str(fold);df['model_component']=kind;df['selected']=df.coefficient.ne(0)&df.feature.ne('(Intercept)');details.append(df)
new=pd.read_csv(R/'final/rscore_full/coefficients.csv');new['Feature']=new.feature.str.removeprefix('rad__');new=new.rename(columns={'coefficient':'New_standardized_coefficient'})
comparison=old.merge(new[['Feature','New_standardized_coefficient']],on='Feature',how='outer');comparison['Old_selected']=comparison.Feature.isin(oldset);comparison['New_selected']=comparison.Feature.isin(sets[-1]);comparison=comparison[comparison.Old_selected|comparison.New_selected|(comparison.Feature=='(Intercept)')]
frequency=pd.DataFrame([{'Feature':v,'Old_selected':v in oldset,**{f'Fold_{i+1}':v in sets[i] for i in range(5)},'Selected_folds':sum(v in s for s in sets[:5]),'Final_selected':v in sets[-1]} for v in sorted(set.union(*sets,oldset))])
metrics=pd.read_csv(ROOT/'tables/performance.csv');rows=[]
for name,label,olddev,oldext in [('CMR','CMR (CatBoost)',.848,.834),('Rscore_full','Rscore alone',.829,.848),('CMR_Rscore_full','CMR + Rscore (CatBoost)',.917,.875)]:
 d=metrics[(metrics.cohort=='development')&(metrics.model==name)].iloc[0];e=metrics[(metrics.cohort=='external_test')&(metrics.model==name)].iloc[0]
 rows.append({'Model':label,'Manuscript_development_AUC':olddev,'New_development_OOF_AUC':d.AUC,'Manuscript_external_AUC':oldext,'New_external_AUC':e.AUC,'New_external_sensitivity':e.Sensitivity,'New_external_specificity':e.Specificity,'New_external_NPV':e.NPV,'Comparison_type':'Historical descriptive comparison; not paired inference; pipeline/label versions differ'})
notes=pd.DataFrame([{'Item':'Historical performance','Source':'JMRI-26-1970-ME-EIC-edits.docx','Note':'Manuscript values transcribed; do not substitute another local workbook version.'},{'Item':'Historical Rscore formula','Source':str(oldpath),'Note':'Local archived standardized 23-feature formula; this workbook reports 217 external events and AUC 0.860499, unlike current 212 events and manuscript AUC 0.848. Feature overlap is with this specific archive, not proof of exact manuscript S5 provenance.'},{'Item':'New performance','Source':str(ROOT/'tables/performance.csv'),'Note':'1059 development / 309 external, 212 external events. Development AUC pools outer-fold predictions; external AUC uses final full-development refit.'},{'Item':'Coefficients','Source':'Each fold outer/cmr_selection and rscore_full/coefficients.csv','Note':'Nonzero coefficients define selected features. Standardization is fitted separately in each training set; coefficients have different fold-specific units.'}])
summary=pd.DataFrame(summary);performance=pd.DataFrame(rows)
with pd.ExcelWriter(ROOT/'tables/五折筛选与新旧结果对照.xlsx') as writer:
 for name,df in [('Performance_comparison',performance),('Fold_selection_summary',summary),('Rscore_feature_frequency',frequency),('Old_new_Rscore_coefficients',comparison),('All_fold_coefficients',pd.concat(details)),('Source_notes',notes)]:df.to_excel(writer,sheet_name=name,index=False)
wb=load_workbook(ROOT/'tables/五折筛选与新旧结果对照.xlsx')
for ws in wb:
 ws.sheet_view.showGridLines=False;ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions;last=ws.max_row
 for row in ws:
  for c in row:c.font=Font(name='Times New Roman',size=12,bold=c.row==1);c.alignment=Alignment(wrap_text=True,vertical='top');c.border=Border(top=Side(style='thin') if c.row==1 else Side(),bottom=Side(style='thin') if c.row in [1,last] else Side())
 for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=26
wb.save(ROOT/'tables/五折筛选与新旧结果对照.xlsx')
t=Tables();t.add('Table C1','Manuscript and new nested-validation performance',['Model','Manuscript development AUC','New development OOF AUC','Manuscript external AUC','New external AUC'],[[r.Model,f'{r.Manuscript_development_AUC:.3f}',f'{r.New_development_OOF_AUC:.3f}',f'{r.Manuscript_external_AUC:.3f}',f'{r.New_external_AUC:.3f}'] for _,r in performance.iterrows()],'Historical descriptive comparison, not a paired estimate of bias. Original values are manuscript-reported. New results use 1059 development patients and 309 external patients (212 events); source/label discrepancies are recorded in the Excel Source_notes sheet.')
t.add('Table C2','Actual selection within each outer training set',['Stage','Selected CMR variables','CMR n','Rscore n','Shape n','Overlap with archived 23'],[[str(r.stage),r.CMR_features,r.CMR_n,r.Rscore_n,r.shape_n,r.overlap_with_archived_23] for _,r in summary.iterrows()],'Final refit uses all development patients and produces the external model. Fold-specific Rscore scores, preprocessing and coefficients were rebuilt in their corresponding training sets. Feature overlap is against the local archived formula, whose exact manuscript-version identity remains to be reconciled.',widths=[2.8,11.3,3,3,3,3])
t.doc.save(ROOT/'tables/五折筛选与新旧结果对照.docx')
print(summary[['stage','CMR_n','Rscore_n','shape_n','overlap_with_archived_23']].to_string(index=False))
