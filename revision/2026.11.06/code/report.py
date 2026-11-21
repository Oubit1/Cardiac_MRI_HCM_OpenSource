"""Build publication-style figures, complete Excel tables and honest response Word.

Safe to run during training. New and archived results remain visibly separated.
Statistical caches depend on patient IDs, outcomes, predictions and threshold.
"""
from common import ROOT, write_json, timestamp, sha256
import hashlib
import fcntl
import json
import os
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact,mannwhitneyu
from sklearn.metrics import roc_curve
from docx import Document
from docx.shared import Inches,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from openpyxl.styles import Font,PatternFill,Alignment
from revision_statistics import metrics_ci,threshold,paired,holm,auc
from report_visuals import build as additional_visuals, paired_text

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
PROTOCOL=json.loads((ROOT/'plans/protocol.json').read_text())
CORE=PROTOCOL.get('active_stage')=='core24h'
ARMS=PROTOCOL['arms']
DISPLAY={'baseline20_full':'mViT: 20-epoch selection window',
         'extended60_full':'mViT: 60-epoch selection window',
         'roi_matched_full':'mViT: ROI, matched training duration',
         'tuned60_full':'mViT: 60 epochs + LR selection',
         'tuned60_roi':'mViT: myocardial ROI',
         'tuned60_full_seed142':'mViT: additional seed 142',
         'tuned60_full_seed242':'mViT: additional seed 242'}


def cached_stats(frame,cut):
    identity=frame[['ID','event','probability']].sort_values('ID').to_csv(index=False)+repr(cut)
    key=hashlib.sha256(('v1:'+identity).encode()).hexdigest()
    path=ROOT/'cache/statistics'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    result=metrics_ci(frame.event.to_numpy(),frame.probability.to_numpy(),cut)
    write_json(path,result)
    return result


def save_figure(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    if (ROOT/'all_training_complete.json').exists():
        fig.savefig(ROOT/'figures'/f'{name}.tiff',dpi=300,bbox_inches='tight',
                    pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


def collect():
    metrics,raw,curves,configs,comparisons=[],[],[],[],[]
    store={}
    # Exact predictions used for the submitted Table 4, not the old-label archive.
    for dataset,name in [('Development OOF','development_oof'),('External test','external_test')]:
        df=pd.read_csv(ROOT/f'inputs/submitted_ft_{name}_predictions.csv')
        dev=pd.read_csv(ROOT/'inputs/submitted_ft_development_oof_predictions.csv')
        for model,col in [('Submitted mViT-FT','probability'),('Submitted mViT-FT + CMR','Foundation+CMR_probability'),
                          ('Submitted mViT-FT + Rscore + CMR','Foundation+Rscore+CMR_probability')]:
            frame=df[['ID','event',col]].rename(columns={col:'probability'})
            cut=threshold(dev.event,dev[col])
            role='Archived submitted result; legacy pipeline limitations retained'
            metrics.append({'Model':model,'Dataset':dataset,'Role':role,**cached_stats(frame,cut)})
            raw.append(frame.assign(Model=model,Dataset=dataset,Role=role))
            store[(model,dataset)]=frame
    thresholds={}
    for arm in ARMS:
        paths=sorted((ROOT/'runs').glob(f'oof_{arm}_fold*/heldout_predictions.csv'))
        if len(paths)==5:
            df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
            assert len(df)==1050 and not df.ID.duplicated().any()
            cut=threshold(df.event,df.probability);thresholds[arm]=cut
            metrics.append({'Model':arm,'Dataset':'Development OOF','Role':'New image-only nested selection',**cached_stats(df,cut)})
            raw.append(df.assign(Model=arm,Dataset='Development OOF',Role='New image-only nested selection'))
            store[(arm,'Development OOF')]=df
            external=ROOT/'runs'/f'final_{arm}'/'external_predictions.csv'
            if external.exists():
                df=pd.read_csv(external)
                assert len(df)==309 and not df.ID.duplicated().any()
                metrics.append({'Model':arm,'Dataset':'External test','Role':'Revision sensitivity; reused external cohort',**cached_stats(df,cut)})
                raw.append(df.assign(Model=arm,Dataset='External test',Role='Revision sensitivity; reused external cohort'))
                store[(arm,'External test')]=df
    write_json(ROOT/'plans/development_locked_thresholds.json',thresholds)
    for directory in sorted((ROOT/'runs').iterdir()):
        if not directory.is_dir():continue
        path=directory/'config.json'
        if path.exists():
            cfg=json.loads(path.read_text());cfg['Run']=directory.name
            cfg['Train_N']=len(cfg.pop('train_ids',[]));cfg['Validation_N']=len(cfg.pop('validation_ids',[]))
            cfg['Complete']=(directory/'complete.json').exists()
            if cfg['Complete']:
                completion=json.loads((directory/'complete.json').read_text())
                cfg['Reused_from']=completion.get('reused_from','')
                cfg['Actual_training_seconds']=completion.get('total_seconds')
            configs.append(cfg)
        path=directory/'history.csv'
        if path.exists():
            df=pd.read_csv(path);df['Run']=directory.name;curves.append(df)
    primary_pairs=[('baseline20_full','extended60_full'),('baseline20_full','tuned60_full'),('tuned60_full','tuned60_roi')]
    if CORE:primary_pairs=[('baseline20_full','extended60_full'),('extended60_full','roi_matched_full')]
    for dataset in ['Development OOF','External test']:
        batch=[]
        for a,b in primary_pairs:
            if (a,dataset) not in store or (b,dataset) not in store:continue
            m=store[(a,dataset)].merge(store[(b,dataset)],on='ID',validate='one_to_one',suffixes=('_a','_b'))
            assert (m.event_a==m.event_b).all()
            batch.append({'Dataset':dataset,'Model_A':a,'Model_B':b,
                          **paired(m.event_a,m.probability_a,m.probability_b)})
        if batch:
            for row,p in zip(batch,holm([r['DeLong_P'] for r in batch])):row['Holm_P']=p
            comparisons.extend(batch)
    fusion=ROOT/'runs'/('final_'+PROTOCOL.get('fusion_source_arm','tuned60_full'))/'external_fixed_fusion_predictions.csv'
    if fusion.exists():
        df=pd.read_csv(fusion)
        for model in df.columns[2:]:
            frame=df[['ID','event',model]].rename(columns={model:'probability'})
            metrics.append({'Model':model,'Dataset':'External test','Role':'Fixed-classifier exploratory fusion; no new OOF claim',**cached_stats(frame,None)})
            raw.append(frame.assign(Model=model,Dataset='External test',Role='Fixed-classifier exploratory fusion'))
        for a,b in [('CMR_fixed','mViT_CMR_fixed'),('Rscore_CMR_fixed','mViT_Rscore_CMR_fixed')]:
            comparisons.append({'Dataset':'External test','Model_A':a,'Model_B':b,**paired(df.event,df[a],df[b])})
    return pd.DataFrame(metrics),pd.concat(raw,ignore_index=True),pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),pd.DataFrame(configs),pd.DataFrame(comparisons),store


def acquisition(store,metrics):
    metadata=pd.read_csv(ROOT/'inputs/acquisition_metadata.csv').drop(columns='event')
    manifest=pd.read_csv(ROOT/'inputs/analysis_manifest.csv')[['cohort','ID','event']]
    meta=metadata.merge(manifest,on=['cohort','ID'],validate='one_to_one')
    meta['resolution_mm']=np.sqrt(meta.pixel_spacing_row_mm*meta.pixel_spacing_col_mm)
    split=meta.loc[meta.cohort=='development','resolution_mm'].median()
    meta['resolution_group']=np.where(meta.resolution_mm.isna(),'missing',np.where(meta.resolution_mm<=split,'<= development median','> development median'))
    associations=[]
    for cohort,group in meta.groupby('cohort'):
        cross=pd.crosstab(group.field_strength_t,group.event).reindex(columns=[0,1],fill_value=0)
        p=float(fisher_exact(cross.to_numpy())[1]) if cross.shape==(2,2) else np.nan
        for field,row in cross.iterrows():
            associations.append({'Cohort':cohort,'Variable':'field_strength_t','Level':field,'N':int(row.sum()),
                                 'Events':int(row[1]),'Fisher_P':p})
        zero=group.loc[group.event==0,'resolution_mm'].dropna();one=group.loc[group.event==1,'resolution_mm'].dropna()
        p=float(mannwhitneyu(zero,one,alternative='two-sided').pvalue)
        for event,values in [(0,zero),(1,one)]:
            associations.append({'Cohort':cohort,'Variable':'resolution_mm','Level':f'event={event}',
                                 'N':len(values),'Median':values.median(),'Q1':values.quantile(.25),
                                 'Q3':values.quantile(.75),'Mann_Whitney_P':p})
    strata=[]
    for (model,dataset),frame in store.items():
        if dataset!='External test' or model in ['Submitted mViT-FT + Rscore + CMR']:continue
        m=frame.merge(meta[meta.cohort=='external_test'].drop(columns='event'),on='ID',validate='one_to_one')
        cut=float(metrics[(metrics.Model==model)&(metrics.Dataset==dataset)].iloc[0].Threshold)
        for variable in ['field_strength_t','resolution_group']:
            for level,group in m.groupby(variable,dropna=False):
                count=group.event.value_counts()
                if len(count)<2 or count.min()<10:
                    result={'N':len(group),'Events':int(group.event.sum()),'CI_status':'insufficient_class_counts'}
                else:result=cached_stats(group,cut)
                strata.append({'Model':model,'Variable':variable,'Level':level,'Resolution_cut_mm':split,
                               'Role':'Exploratory; centre/scanner confounding not separated',**result})
    return pd.DataFrame(strata),pd.DataFrame(associations)


def figures(history,store):
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for j,name in enumerate(['Submitted mViT-FT','Submitted mViT-FT + CMR','Submitted mViT-FT + Rscore + CMR']):
        frame=store[(name,'External test')]
        fpr,tpr,_=roc_curve(frame.event,frame.probability)
        axes[0].plot(fpr,tpr,color=COLORS[j],label=f"{name.replace('Submitted ','')}: {auc(frame.event,frame.probability):.3f}")
        bins=pd.qcut(frame.probability,10,duplicates='drop')
        points=frame.groupby(bins,observed=True).agg(predicted=('probability','mean'),observed=('event','mean'))
        axes[1].plot(points.predicted,points.observed,'o-',color=COLORS[j],markersize=4,label=name.replace('Submitted ',''))
    for ax in axes:
        ax.plot([0,1],[0,1],ls='--',color='#999999',lw=.8);ax.set(xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7,loc='lower right')
    axes[0].set(xlabel='1 - Specificity',ylabel='Sensitivity',title='Submitted external predictions: ROC')
    axes[1].set(xlabel='Mean predicted probability',ylabel='Observed event fraction',title='Submitted external predictions: calibration')
    fig.tight_layout();save_figure(fig,'Figure_S0_submitted_reference')
    if not history.empty:
        fig,axes=plt.subplots(2,5,figsize=(14,5.7),sharex=True,sharey='row')
        for fold in range(1,6):
            h=history[history.Run==f'inner_full_fold{fold}_lr1e-05']
            for row,metric in [(0,'loss'),(1,'auc')]:
                ax=axes[row,fold-1]
                ax.axvline(20,color='#666666',ls='--',lw=.8)
                ax.axvspan(.5,2.5,color='#dddddd',alpha=.4)
                if not h.empty:
                    ax.plot(h.epoch,h[f'train_{metric}'],color=COLORS[0],label='Training (evaluation mode)')
                    ax.plot(h.epoch,h[f'validation_{metric}'],color=COLORS[1],label='Inner validation')
                else:ax.text(.5,.5,'Pending',transform=ax.transAxes,ha='center')
                ax.set_xlim(.5,60.5);ax.grid(alpha=.15)
                if row==0:ax.set_title(f'Fold {fold}')
                else:ax.set_xlabel('Epoch')
        axes[0,0].set_ylabel('Binary cross-entropy');axes[1,0].set_ylabel('AUC')
        handles,labels=axes[0,0].get_legend_handles_labels()
        if handles:fig.legend(handles,labels,loc='lower center',ncol=2)
        fig.suptitle('mViT training-duration sensitivity (encoder learning rate = 1e-5)')
        fig.tight_layout(rect=[0,.06,1,.95]);save_figure(fig,'Figure_S1_training_curves')
        learning_rates=PROTOCOL['training']['encoder_lrs']
        grid=np.full((len(learning_rates),5),np.nan)
        for j,lr in enumerate(learning_rates):
            for i in range(5):
                h=history[history.Run==f'inner_full_fold{i+1}_lr{lr:g}']
                if not h.empty:grid[j,i]=h.validation_auc.max()
        fig,ax=plt.subplots(figsize=(7,3))
        im=ax.imshow(np.ma.masked_invalid(grid),cmap='Blues',vmin=.5,vmax=1.,aspect='auto')
        ax.set_xticks(range(5),[f'Fold {i}' for i in range(1,6)])
        ax.set_yticks(range(len(learning_rates)),[f'{lr:g}' for lr in learning_rates]);ax.set_ylabel('Encoder learning rate')
        for j in range(len(learning_rates)):
            for i in range(5):ax.text(i,j,f'{grid[j,i]:.3f}' if np.isfinite(grid[j,i]) else 'Pending',ha='center',va='center',color='black')
        fig.colorbar(im,ax=ax,label='Best observed inner-validation AUC')
        ax.set_title('Fixed-LR checkpoint selection (partial trajectories marked in workbook)' if CORE else 'Learning-rate sensitivity (partial trajectories marked in workbook)')
        fig.tight_layout();save_figure(fig,'Figure_S2_learning_rates')
    for dataset,suffix in [('Development OOF','OOF'),('External test','external')]:
        available=[(arm,store[(arm,dataset)]) for arm in ARMS[:4] if (arm,dataset) in store]
        if not available:continue
        fig,ax=plt.subplots(figsize=(5.2,5))
        for (name,frame),color in zip(available,COLORS):
            fpr,tpr,_=roc_curve(frame.event,frame.probability)
            ax.plot(fpr,tpr,color=color,label=f'{DISPLAY[name]}: {auc(frame.event,frame.probability):.3f}')
        ax.plot([0,1],[0,1],color='#999999',ls='--',lw=.8)
        ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=dataset,xlim=(0,1),ylim=(0,1))
        ax.legend(loc='lower right',fontsize=7);fig.tight_layout();save_figure(fig,f'Figure_S3_ROC_{suffix}')


def word(metrics,comparisons,configs):
    final=(ROOT/'all_training_complete.json').exists()
    doc=Document();sec=doc.sections[0]
    sec.top_margin=sec.bottom_margin=Inches(.7)
    doc.styles['Normal'].font.name='Arial';doc.styles['Normal'].font.size=Pt(10)
    doc.add_paragraph('Response to reviewers: mViT supplementary experiments')
    doc.add_paragraph(('Completed analysis package' if final else 'WORKING DRAFT — NEW EXPERIMENTS IN PROGRESS')+' | '+timestamp())
    doc.add_paragraph('This document addresses only the mViT-related comments. Pending analyses are explicitly identified. Manuscript page/line references must be added after integration into the annotated EIC manuscript. No claim is made that the previously evaluated external cohort is a new, untouched test set.')
    if CORE:
        doc.add_paragraph('Scope amendment: at the authors\' request for a roughly 24-hour core delivery, the supplement prioritizes five-fold 20-versus-60-epoch comparisons and an ROI arm matched to the full-image training duration. The learning-rate grid, separate ROI tuning, and additional-seed analyses are deferred and are not claimed complete. The amendment was made after the first inner trajectory reached epoch 50, before any new outer-held-out or external evaluation. SDPA and removal of gradient recomputation accelerate the same model; mixed-precision numerical differences and the resumed execution boundary are recorded in the audit and per-run execution logs.')
    def response(title,text):
        p=doc.add_paragraph();p.add_run(title).bold=True;doc.add_paragraph(text)
    def result(model,dataset):
        x=metrics[(metrics.Model==model)&(metrics.Dataset==dataset)]
        if x.empty:return 'pending'
        r=x.iloc[0];return f"{r.AUC:.3f} (95% CI {r.AUC_CI_low:.3f}–{r.AUC_CI_high:.3f})"
    response('Data and version reconciliation',
      'The submitted Table 4 mViT-FT AUCs were reproduced from the archived patient-level predictions: development OOF AUC 0.792735 and external AUC 0.801692. An older archive contains identical image-only probabilities but six different external outcome labels, explaining its AUC of 0.836632. The revision uses the current labels, which agree with the clinical and radiomics spreadsheets. The label discrepancy is documented separately and is not presented as a change in model performance. Nearby configuration files do not by themselves establish the configuration of the submitted prediction files.')
    audit=ROOT/'audit/submitted_checkpoint_reproduction.json'
    if audit.exists():
        proof=json.loads(audit.read_text())
        response('Submitted checkpoint reproduction audit',
          f"The candidate archived checkpoint was evaluated on all {proof['n']} external cine inputs using current labels. Its recorded training duration is {proof['locked_epochs']} epochs. Maximum absolute probability difference from the submitted predictions was {proof['max_abs_probability_difference']:.6g}; tolerance was {proof['tolerance']}. Match within tolerance: {proof['matches_within_tolerance']}. The reproduced AUC was {proof['reproduced_auc']:.6f}. The checkpoint hash and individual predictions are retained in the audit folder. This is a provenance check, not an additional model-selection step.")
    response('Reviewer 1 — Training duration and optimization',
      'We agree that a 20-epoch upper limit alone cannot establish adequate optimization. The amended core revision experiment trains the original encoder learning rate (1e-5) for 60 epochs without early stopping in each of five outer folds. This tests training duration at a fixed learning rate; it is not a comprehensive hyperparameter search. Within each outer training fold, a separate inner-validation subset selects checkpoints within either the first 20 or all 60 epochs. The selected duration is then used to refit the model on the complete outer training fold before predicting the outer held-out patients. Training and inner-validation AUC and loss are recorded at each epoch; training evaluation uses augmentation-free inputs. New development OOF AUCs: baseline20_full '+result('baseline20_full','Development OOF')+'; extended60_full '+result('extended60_full','Development OOF')+'. The workbook reports paired differences and confidence intervals when all five folds are complete. A nonsignificant comparison is not interpreted as proof of equivalence or global optimality.')
    selections=ROOT/'plans/completed_fold_selections.json'
    if selections.exists():
        chosen=json.loads(selections.read_text())
        if len(chosen)==5:
            doc.add_paragraph('The locked outer-training refit durations for folds 1–5 were '+
                ', '.join(str(x['baseline_epoch']) for x in chosen)+' epochs for the 20-epoch window and '+
                ', '.join(str(x['extended_epoch']) for x in chosen)+' epochs for the 60-epoch window. Raw selected checkpoint epochs and the minimum-three-epoch refit rule are recorded separately in the Fold_checkpoint_selection table.')
            if all(x['baseline_epoch']==x['extended_epoch'] for x in chosen):
                doc.add_paragraph('The two selection windows yielded identical refit configurations in all five folds. Those fits share checkpoints and are not independent model replicates. Extending the selection window therefore did not change the selected refit durations in this experiment; this does not establish global optimality or equivalence across alternative training strategies.')
    response('Reviewer 1 — AdamW and fine-tuning strategy',
      'The revision fixes AdamW beta1=0.9, beta2=0.999, epsilon=1e-8, weight decay=1e-4, and dropout=0.25. The encoder is frozen for the first two epochs and then fully unfrozen. The attention/classification modules use a learning rate of 1e-4; the encoder learning rate is fixed at 1e-5 in the core experiment. The originally planned learning-rate grid is deferred. The lower encoder rate is intended to limit disruption of pretrained representations, whereas the newly initialized modules require task adaptation. These are design motivations, not claims of optimality. Four patient gradients are accumulated, and gradient norms are clipped at 1.0. Slice microbatch size is eight and is distinct from patient batch size. No cosine scheduler or learning-rate ramp is claimed. Additional-seed experiments are deferred; the core results do not establish seed robustness.')
    response('Reviewer 1 — Which model generated the external AUC?',
      'For the new revision experiments, the final model is refitted on all 1,050 development patients from the same pretrained initialization. Its encoder learning rate is fixed at 1e-5, and its duration is the median of the fold-selected epochs, with a minimum of three epochs. The ROI arm uses the same duration as its paired full-image 60-epoch-selection arm. The baseline and extended fixed-learning-rate arms use their respective median durations. All final configurations are written before new external inference. No fold model or random seed is chosen by external AUC. The exact configuration and checkpoint provenance of the original submitted model are documented separately from this new procedure.')
    response('Reviewer 1 — Universal approximation and incremental value',
      'Universal approximation concerns representational capacity under specified assumptions; it does not guarantee that a fixed mViT architecture trained with finite samples and a finite optimization budget will recover a handcrafted radiomics mapping or achieve higher external discrimination. Expert myocardial contours also supply localization information that is not explicitly provided to the whole-image mViT. Lower performance of mViT+CMR than Rscore+CMR and lack of improvement after adding mViT to Rscore+CMR are separate observations. Neither establishes biological redundancy. We interpret the results as properties of the evaluated inputs, training and fusion procedures, rather than intrinsic superiority of radiomics over optimized deep learning.')
    response('Reviewer 3, comment 5 — Benchmark fairness',
      'The revision includes a paired exploratory cine-mask localization experiment. Within each fold, ROI and whole-image refits use the same learning rate, initialization seed, training patients and selected duration; the duration is selected using whole-image inner validation. This is a matched-budget localization comparison, not independently optimized ROI performance. A fixed crop for each slice is formed from the mask union over the cardiac cycle with a 20% margin on each side. The crop uses no LGE image or LGE-derived region. Patients are retained regardless of mask QC; flagged or unavailable masks trigger the prespecified whole-image fallback, whose frequency is reported. The masks are existing binary cine masks with incompletely documented generation provenance and are not claimed to be expert manual ground truth. Accordingly, this experiment examines sensitivity to localization but cannot establish equal supervision or fully resolve benchmark fairness. Whole-image 60-epoch-selection external AUC: '+result('extended60_full','External test')+'; matched ROI external AUC: '+result('roi_matched_full','External test')+'.')
    response('Reviewer 1 — Field strength and spatial resolution',
      'The workbook reports cohort-specific field-strength distributions, their association with LGE status, and in-plane pixel-spacing distributions by outcome. External model performance is stratified by field strength and by the development-derived median in-plane spacing. All operating thresholds remain those selected from development OOF predictions. Patient-stratified bootstrap intervals quantify uncertainty. These observational strata do not separate centre, protocol and field-strength effects; nonsignificant subgroup differences would not establish equivalence.')
    response('Reviewer 1 — Development versus external AUC',
      'The submitted image-only values are 0.793 and 0.802, respectively. OOF and external estimates arise from different patients and different model-training sample sizes. Sampling variation and case-mix differences can yield a numerically higher external AUC. This pattern alone neither establishes superior transportability nor demonstrates undertraining. The revised experiments provide direct optimization evidence and uncertainty estimates.')
    response('Editorial Board — Validation independence and external reuse',
      'New image-only experiments keep outer held-out patients outside supervised adaptation, learning-rate selection and checkpoint selection. Inner validation selects configurations, followed by an outer-training refit. This does not retrospectively establish complete nesting of the legacy downstream pipeline. The user-confirmed CMR script selects variables on the full development cohort; precomputed global OOF Rscores do not automatically confer independence on subsequent fusion CV. New representation-fusion sensitivity uses a classifier fixed in advance and reports external results only, with no new fusion OOF claim. The external cohort has previously been examined, so revised external results are described as sensitivity analyses, not fresh confirmatory validation. Historical use of the external-tuning utility must be checked by the authors; the existence of that utility alone does not establish that it was used for the submitted model.')
    response('Methods details requiring accurate wording',
      'The implemented temporal operation is uniform rounded-index frame selection, not linear temporal interpolation. The classification head is a two-layer MLP after layer normalization, rather than a single linear layer. The archived single-frame ablation repeats the first cine frame; it should be called ED-repeat only if end-diastolic phase provenance is verified. Sensitivity and specificity comparisons must use development-derived operating rules rather than externally optimized thresholds.')
    temporal=ROOT/'tables/Temporal_archived_metrics.csv'
    if temporal.exists():
        temp=pd.read_csv(temporal)
        values='; '.join(f"{r.Model}, {r.Dataset}: AUC {r.AUC:.3f}" for r in temp.itertuples())
        response('Supplementary temporal information analysis',
          'Archived full-cine and repeated-first-frame predictions were re-evaluated with the current endpoint labels and matched by patient ID. No new training was performed for this analysis. Apart from temporal mode and administrative fields, the archived run configurations agree. '+values+'. These results assess the contribution of temporal input in the archived implementation; they do not prove optimal training or equal anatomical supervision. Frame 0 has not been independently verified as end-diastole.')
    response('Status and integration instructions',
      ('All amended core training stages are complete. Learning-rate search, separate ROI tuning and additional seeds remain deferred. Review the paired estimates and manuscript scope before submission.' if final else
       'New training is not yet complete. Do not submit this working draft as if all experiments had finished. The queue refreshes this document and the workbook as runs finish; FINAL_COMPLETE.json is created only after the final analyses and report succeed.'))
    if not comparisons.empty:
        response('Quantitative comparisons supporting the responses',
                 'Differences are calculated on matched patients. Confidence intervals use 2,000 event-stratified patient bootstrap resamples; the primary image comparisons receive Holm adjustment within each dataset. No difference or a confidence interval including zero is not interpreted as proof of equivalence. Identical reused fits are not independent replicates.')
        for paragraph in paired_text(comparisons,DISPLAY):doc.add_paragraph(paragraph)
    if final and CORE:
        response('Interpretation of the completed duration and ROI experiments',
            'The extended selection window did not yield a consistent out-of-sample gain. Four of five folds selected their best inner-validation checkpoint within the first 20 epochs; fold 4 selected epoch 31 when the window was extended. Thus later improvement can occur in an individual fold, and the data do not justify claiming that 20 epochs is universally sufficient. The paired external differences for extended versus baseline training and ROI versus matched whole-image training both have confidence intervals spanning zero. These experiments address two concrete optimization and input-localization concerns, but do not exclude gains from other learning rates, tuning strategies, or better-validated myocardial masks.')
        ext=metrics[(metrics.Model=='extended60_full')&(metrics.Dataset=='External test')].iloc[0]
        operating=(f"At the threshold locked from development OOF predictions ({ext.Threshold:.6f}), the extended-window model had external sensitivity {ext.Sensitivity:.3f} and specificity {ext.Specificity:.3f}, with TP={int(ext.TP)}, FP={int(ext.FP)}, TN={int(ext.TN)} and FN={int(ext.FN)}. ")
        if ext.TN+ext.FN==0:
            operating+='All external patients were classified positive, so NPV is undefined and this operating point provides no negative triage group. '
        operating+='This illustrates that discrimination alone does not establish useful operating-point performance. The results are consistent with limited transfer of the pooled-OOF threshold to the full-development refit; this explanation is not a proven causal mechanism. The threshold was not retuned on the external cohort. Joint calibration estimates and their confidence intervals are reported in the workbook.'
        response('Operating-point limitation and calibration',operating)
        response('Model-label clarification and incremental value',
            'In the verified submitted Table 4, external AUC is 0.786 for mViT-Frozen + CMR and 0.821 for mViT-FT + CMR; the approximately 0.82 result should therefore be attributed to the fine-tuned combination. The new fixed-classifier fusion sensitivity also does not support incremental benefit from adding the evaluated mViT representation. This is a result for the specified representation and fusion procedure, not proof of intrinsic radiomics superiority or biological redundancy; the new fusion analysis remains exploratory and external-only.')
    doc.add_paragraph('Performance summary (joint calibration coefficients and full operating metrics are in Excel):')
    table=doc.add_table(rows=1,cols=5);table.style='Table Grid'
    for cell,label in zip(table.rows[0].cells,['Model','Dataset','N','AUC [95% CI]','Brier']):cell.text=label
    for _,r in metrics.iterrows():
        values=[DISPLAY.get(r.Model,r.Model),r.Dataset,str(int(r.N)),f'{r.AUC:.3f} [{r.AUC_CI_low:.3f}, {r.AUC_CI_high:.3f}]',f'{r.Brier:.3f}']
        for cell,value in zip(table.add_row().cells,values):cell.text=value
    for path in sorted((ROOT/'figures').glob('Figure_*.png')):
        doc.add_page_break()
        doc.add_paragraph(path.stem.replace('_',' '));doc.add_picture(str(path),width=Inches(6.4))
        doc.add_paragraph(figure_caption(path.stem))
    doc.add_paragraph('References: Loshchilov and Hutter, Decoupled Weight Decay Regularization, https://arxiv.org/abs/1711.05101; Hornik, Approximation capabilities of multilayer feedforward networks, https://doi.org/10.1016/0893-6080(91)90009-T. AdamW defaults: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.')
    doc.save(ROOT/'response/mViT_response.docx')


def figure_caption(name):
    captions={
        'S0':'Archived submitted external predictions, evaluated using the current endpoint labels. These are not newly trained models.',
        'S1':'Augmentation-free training and inner-validation loss and AUC at each epoch. The dashed line marks epoch 20; shading marks the two frozen-encoder epochs. Inner-validation curves select checkpoints and are not outer-held-out performance estimates.',
        'S2':'Highest observed inner-validation AUC by fold at the fixed encoder learning rate. Incomplete trajectories remain explicitly marked in the workbook; this core-stage figure is not a completed learning-rate search.',
        'S3':'ROC curves for completed new image-model evaluations. Development predictions are assembled only after all five outer folds; external predictions follow development-only configuration locking.',
        'S4':'Descriptive calibration using up to ten equal-frequency probability bins. Bars are pointwise 95% Wilson intervals for the observed bin event fraction, not confidence bands for a fitted calibration function. Bin counts and coordinates are in Calibration_bins; joint calibration intercept and slope with bootstrap intervals are in Performance.',
        'S5':'AUC estimates with 95% event-stratified patient-bootstrap intervals (2,000 resamples). Separate intervals do not constitute a paired significance test.',
        'S6':'Paired AUC differences are Model B minus Model A, using matched patients and 2,000 stratified bootstrap resamples. Zero denotes no observed difference. Adjusted and unadjusted P values are distinguished in Paired_comparisons; identical reused fits are not independent replicates.',
        'S7':'Exploratory external subgroup AUC with patient-bootstrap 95% intervals. Counts show patients and events. Resolution is the geometric mean of in-plane pixel spacing, dichotomized at the development median. Groups with fewer than ten cases of either outcome are not estimated. Centre, scanner and acquisition effects are not separated.'}
    return captions.get(name.split('_')[1],'See source data and analysis role in the accompanying workbook.')


def main():
    for folder in ['figures','tables','response','cache/statistics']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
    metrics,raw,history,configs,comparisons,store=collect()
    strata,associations=acquisition(store,metrics)
    figures(history,store)
    extra=additional_visuals(metrics,comparisons,strata,store,ARMS,DISPLAY,COLORS,save_figure)
    sheets={'Performance':metrics,'Paired_comparisons':comparisons,'Acquisition_subgroups':strata,
            'Acquisition_LGE_association':associations,'Training_runs':configs,'Epoch_history':history,
            'Patient_predictions':raw}
    sheets.update(extra)
    selections=[]
    for fold in range(1,6):
        p=ROOT/'runs'/f'inner_full_fold{fold}_lr1e-05'/'complete.json'
        if not p.exists():continue
        best=json.loads(p.read_text())['best']
        selections.append({'Fold':fold,'Checkpoint_epoch_20':best['20']['epoch'],
                           'Checkpoint_epoch_60':best['60']['epoch'],'Inner_AUC_20':best['20']['auc'],
                           'Inner_AUC_60':best['60']['auc'],'Refit_epoch_20':max(3,best['20']['epoch']),
                           'Refit_epoch_60':max(3,best['60']['epoch']),
                           'ROI_refit_epoch':max(3,best['60']['epoch'])})
    sheets['Fold_checkpoint_selection']=pd.DataFrame(selections)
    sheets['Figure_legends']=pd.DataFrame([{'Figure':p.stem,'Caption':figure_caption(p.stem),
                  'Formats':'PNG + PDF; TIFF (300 dpi) upon core completion'} for p in sorted((ROOT/'figures').glob('Figure_*.png'))])
    sheets['Reviewer_evidence_map']=pd.DataFrame([
        {'Comment':'R1: 20 epochs and training curves','Evidence':'Five complete 60-epoch trajectories; paired checkpoint windows','Figures':'S1, S5, S6','Tables':'Fold_checkpoint_selection; Epoch_history; Paired_comparisons'},
        {'Comment':'R1: optimizer and final test model','Evidence':'Fixed AdamW settings; inner-only checkpoint selection; full-development refit','Figures':'S1','Tables':'Protocol; Training_runs; Execution_sessions'},
        {'Comment':'R3-5: anatomical input fairness','Evidence':'Whole-image versus ROI with matched training budget; mask provenance limitation','Figures':'S3, S5, S6','Tables':'Performance; Paired_comparisons; ROI_QC'},
        {'Comment':'R1: scanner field strength and in-plane spacing','Evidence':'Acquisition distribution, outcome association and exploratory external subgroups','Figures':'S7','Tables':'Acquisition_LGE_association; Acquisition_subgroups'},
        {'Comment':'R3-8: external calibration (mViT portion)','Evidence':'Joint calibration intercept/slope with 95% CI; descriptive calibration bins','Figures':'S4','Tables':'Performance; Calibration_bins'},
        {'Comment':'R1: temporal information and incremental value','Evidence':'Archived full-cine/first-frame comparison; fixed-classifier external fusion sensitivity','Figures':'S6 for new fusion when available','Tables':'Temporal_archived_metrics; Temporal_archived_comparisons; Performance; Paired_comparisons'},
        {'Comment':'Editorial Board: validation independence','Evidence':'Image-only nested selection, previous external use and legacy fusion limitations disclosed','Figures':'Not applicable','Tables':'Protocol; Training_runs; Deferred_experiments'}])
    publication=[]
    for _,row in metrics.iterrows():
        entry={'Model':DISPLAY.get(row.Model,row.Model),'Dataset':row.Dataset,'N':int(row.N),'Events':int(row.Events),'Role':row.Role}
        for name in ['AUC','Sensitivity','Specificity','NPV','PPV','Brier','Calibration_intercept_joint','Calibration_slope_joint']:
            value=row.get(name,np.nan);low=row.get(name+'_CI_low',np.nan);high=row.get(name+'_CI_high',np.nan)
            entry[name+' [95% CI]']=(f'{value:.3f} [{low:.3f}, {high:.3f}]' if np.isfinite(low) and np.isfinite(high)
                                      else f'{value:.3f} [CI not estimable]' if np.isfinite(value) else 'Not estimated')
        publication.append(entry)
    sheets['Publication_summary']=pd.DataFrame(publication)
    matrix=[]
    for fold in range(1,6):
        for lr in PROTOCOL['training']['encoder_lrs']:matrix.append({'Run':f'inner_full_fold{fold}_lr{lr:g}','Stage':'Inner selection','Fold':fold,'Planned_LR':lr,'Epoch_cap':60,'Mode':'full'})
        if not CORE:matrix.append({'Run':f'inner_roi_fold{fold}','Stage':'Inner ROI selection','Fold':fold,'Planned_LR':'paired with selected full-image LR','Epoch_cap':60,'Mode':'roi'})
        for arm in ARMS:matrix.append({'Run':f'oof_{arm}_fold{fold}','Stage':'Outer-training refit','Fold':fold,'Planned_LR':1e-5 if CORE else 'development-selected','Epoch_cap':'locked from inner validation; ROI matched to full image' if CORE else 'locked from inner validation','Mode':'roi' if 'roi' in arm else 'full'})
    for arm in ARMS:matrix.append({'Run':f'final_{arm}','Stage':'Full-development refit','Fold':'all','Planned_LR':1e-5 if CORE else 'locked across folds','Epoch_cap':'median selected epoch','Mode':'roi' if 'roi' in arm else 'full'})
    for entry in matrix:
        d=ROOT/'runs'/entry['Run'];entry['Status']='complete' if (d/'complete.json').exists() else 'started' if (d/'config.json').exists() else 'pending'
        if (d/'complete.json').exists():
            completed=json.loads((d/'complete.json').read_text())
            entry['Reused_from']=completed.get('reused_from','')
            if entry['Reused_from']:entry['Status']='complete (identical refit reused)'
        if (d/'history.csv').exists():entry['Last_completed_epoch']=int(pd.read_csv(d/'history.csv').epoch.max())
    sheets['Experiment_matrix']=pd.DataFrame(matrix)
    if CORE:sheets['Deferred_experiments']=pd.DataFrame({'Experiment':PROTOCOL['deferred_experiments'],'Status':'Deferred; not executed in core scope'})
    execution=ROOT/'audit/acceleration_benchmark.json'
    if execution.exists():sheets['Execution_benchmark']=pd.DataFrame(json.loads(execution.read_text())['results'])
    sessions=[]
    for path in sorted((ROOT/'runs').glob('*/execution_sessions.json')):
        for entry in json.loads(path.read_text()):
            sessions.append({'Run':path.parent.name,**{k:v for k,v in entry.items() if k!='execution'},
                             **entry['execution']})
    if sessions:sheets['Execution_sessions']=pd.DataFrame(sessions)
    reference=pd.read_csv(ROOT/'inputs/Table_4_unified_comparison.csv');sheets['Submitted_Table4']=reference
    for name in ['Temporal_archived_metrics','Temporal_archived_comparisons','Temporal_archived_predictions']:
        source=ROOT/'tables'/f'{name}.csv'
        if source.exists():sheets[name]=pd.read_csv(source)
    protocol=json.loads((ROOT/'plans/protocol.json').read_text())
    sheets['Protocol']=pd.DataFrame([{'Item':k,'Value':json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else str(v)} for k,v in protocol.items()])
    for key,path in [('Label_audit','audit/external_label_audit.csv'),('ROI_QC','audit/cache_slice_index.csv'),('Version_AUC_audit','audit/all_archived_prediction_aucs.csv'),('Rscore_source_audit','audit/rscore_source_alignment.csv')]:
        if (ROOT/path).exists():sheets[key]=pd.read_csv(ROOT/path)
    for name,df in sheets.items():
        df.to_csv(ROOT/'tables'/f'{name}.csv',index=False)
    sheets['Read_me']=pd.DataFrame({'Note':[
        'Archived and new model results are separated by Role. Pending new arms are absent from Performance, not silently filled.',
        'N=1050 development and N=309 external only after all relevant predictions are present.',
        'CI: 2000 event-stratified patient bootstrap resamples; threshold fixed from development OOF, not reselected in bootstrap.',
        'Calibration intercept and slope are estimated jointly. They are not offset-only calibration-in-the-large.',
        'Subgroup metrics require >=10 positive and >=10 negative patients; otherwise explicitly not estimable.',
        'New fusion: prespecified fixed classifier, external only, no operating-threshold or independent OOF claim.',
        'Existing mask provenance incomplete; ROI exploratory, whole-image fallback retained and reported.',
        'Core24h amendment retains five 60-epoch fixed-LR trajectories. ROI duration is matched to full images; LR grid and extra seeds are deferred.',
        'Identical refit reuse is not an independent replicate. Actual_training_seconds excludes reused compute; Epoch_history retains the source history.',
        'Execution_benchmark uses one development patient with eight slices and loss scaling 1024 for kernel/gradient checks; it is not a clinical performance result.',
        'Training curves use augmentation-free training evaluation; train_augmented_loss is the optimization objective.',
        'Paired delta-AUC is Model B minus Model A. Holm adjustment applies to the prespecified primary image comparisons.',
        'AUC source discrepancy (0.837 vs 0.802) arose from six changed external labels with identical image-only probabilities.',
        'External cohort was previously evaluated. New external results are revision sensitivities, not untouched confirmation.',
        'Full-run completion requires FINAL_COMPLETE.json; Word is a working draft until then.']})
    dest=ROOT/'tables/mViT_revision_results.xlsx';temp=dest.with_name('mViT_revision_results.tmp.xlsx')
    with pd.ExcelWriter(temp,engine='openpyxl') as writer:
        for name,frame in sheets.items():
            data=frame.copy()
            for column in data:
                data[column]=data[column].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
            data.to_excel(writer,sheet_name=name,index=False)
            ws=writer.book[name];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            ws.sheet_properties.pageSetUpPr.fitToPage=True
            ws.page_setup.orientation='landscape';ws.page_setup.paperSize=ws.PAPERSIZE_A4
            ws.page_setup.fitToWidth=1;ws.page_setup.fitToHeight=0;ws.print_title_rows='1:1'
            for cell in ws[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='234E70');cell.alignment=Alignment(wrap_text=True,vertical='center')
            ws.row_dimensions[1].height=32
            for j,col in enumerate(data.columns,1):
                letter=ws.cell(1,j).column_letter
                ws.column_dimensions[letter].width=min(45,max(14,len(str(col))+2))
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value,float):
                        column=str(data.columns[cell.column-1])
                        if column.endswith('_P') or column in ['DeLong_P','Holm_P'] or 'lr' in column.lower():cell.number_format='0.000E+00'
                        elif column in ['N','Events','TP','FP','TN','FN','Train_N','Validation_N','epoch','epochs','seed','Bootstrap_replicates'] or 'valid_bootstraps' in column:cell.number_format='0'
                        else:cell.number_format='0.0000'
    temp.replace(dest)
    word(metrics,comparisons,configs)
    if (ROOT/'all_training_complete.json').exists():
        for arm in ARMS:
            for dataset,n in [('Development OOF',1050),('External test',309)]:
                assert (arm,dataset) in store and len(store[(arm,dataset)])==n
                bins=extra['Calibration_bins']
                assert int(bins[(bins.Model==arm)&(bins.Dataset==dataset)].N.sum())==n
        required=['Figure_S1_training_curves','Figure_S3_ROC_OOF','Figure_S3_ROC_external',
                  'Figure_S4_calibration_OOF','Figure_S4_calibration_external',
                  'Figure_S5_AUC_OOF','Figure_S5_AUC_external','Figure_S6_paired_differences']
        for name in required:
            for ext in ['png','pdf','tiff']:assert (ROOT/'figures'/f'{name}.{ext}').is_file()
        write_json(ROOT/'audit/final_report_validation.json',{'passed':True,'checked_at':timestamp(),
                   'scope':'Amended core protocol only; deferred experiments remain listed',
                   'checks':['all core OOF/external patient counts','calibration bin counts',
                             'required figures in PNG/PDF/TIFF','Excel and Word successfully generated']})
    write_json(ROOT/'response/report_status.json',{'generated_at':timestamp(),
               'training_complete':(ROOT/'all_training_complete.json').exists(),
               'performance_rows':len(metrics),'completed_new_arms':metrics.loc[metrics.Role.str.startswith('New'),'Model'].tolist()})
    print('Report updated:',dest)


if __name__=='__main__':
    with (ROOT/'report.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
