"""Re-evaluate archived full-cine / first-frame predictions with current labels."""
from common import ROOT,write_json
import json
import pandas as pd
from revision_statistics import metrics_ci,paired,threshold


def main():
    metrics=[];pairs=[];raw=[]
    full_dev=pd.read_csv(ROOT/'inputs/submitted_ft_development_oof_predictions.csv')
    single_dev=pd.read_csv(ROOT/'inputs/archived_first_frame_development.csv')
    for dataset,full_file,single_file in [
        ('Development OOF','submitted_ft_development_oof_predictions.csv','archived_first_frame_development.csv'),
        ('External test','submitted_ft_external_test_predictions.csv','archived_first_frame_external.csv')]:
        full=pd.read_csv(ROOT/'inputs'/full_file)[['ID','event','probability']]
        single=pd.read_csv(ROOT/'inputs'/single_file)[['ID','event','probability']]
        merged=full.merge(single,on='ID',validate='one_to_one',suffixes=('_full','_first'))
        for name,col,dev in [('Full cine','probability_full',full_dev),('Repeated first frame','probability_first',single_dev)]:
            # All analyses use canonical endpoint labels, not the obsolete archive labels.
            training=dev[['ID','probability']].merge(full_dev[['ID','event']],on='ID',validate='one_to_one')
            cut=threshold(training.event,training.probability)
            metrics.append({'Dataset':dataset,'Model':name,'Role':'Archived temporal sensitivity, current endpoint labels',
                            **metrics_ci(merged.event_full,merged[col],cut)})
            raw.append(pd.DataFrame({'ID':merged.ID,'event':merged.event_full,'probability':merged[col],
                                     'Dataset':dataset,'Model':name}))
        pairs.append({'Dataset':dataset,'Model_A':'Repeated first frame','Model_B':'Full cine',
                      'Changed_archive_labels':int((merged.event_full!=merged.event_first).sum()),
                      **paired(merged.event_full,merged.probability_first,merged.probability_full)})
    pd.DataFrame(metrics).to_csv(ROOT/'tables/Temporal_archived_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(ROOT/'tables/Temporal_archived_comparisons.csv',index=False)
    pd.concat(raw,ignore_index=True).to_csv(ROOT/'tables/Temporal_archived_predictions.csv',index=False)
    a=json.loads((ROOT/'inputs/archived_full_run_config.json').read_text())
    b=json.loads((ROOT/'inputs/archived_first_frame_run_config.json').read_text())
    differences={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    write_json(ROOT/'audit/temporal_config_comparison.json',{
        'differences':differences,'phase_statement':'The implementation repeats frame 0; ED status not independently verified.',
        'new_training':False,'external_role':'Reused-cohort post hoc archival sensitivity, not new confirmation'})


if __name__=='__main__':main()
