"""Additional descriptive plots and their exact source tables; no model selection."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build(metrics, comparisons, strata, store, arms, display, colors, save):
    bins=[]
    for dataset,suffix in [('Development OOF','OOF'),('External test','external')]:
        available=[(arm,store[(arm,dataset)]) for arm in arms if (arm,dataset) in store]
        if not available:continue
        fig,axes=plt.subplots(1,len(available),figsize=(4.1*len(available),4),squeeze=False)
        for ax,(arm,frame),color in zip(axes[0],available,colors):
            groups=(pd.qcut(frame.probability,10,duplicates='drop') if frame.probability.nunique()>1
                    else pd.Series(0,index=frame.index))
            points=[]
            for number,(_,g) in enumerate(frame.groupby(groups,observed=True),1):
                n=len(g); p=float(g.event.mean()); z=1.959963984540054
                centre=(p+z*z/(2*n))/(1+z*z/n)
                half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
                points.append({'Model':arm,'Dataset':dataset,'Bin':number,'N':n,'Events':int(g.event.sum()),
                               'Mean_prediction':float(g.probability.mean()),'Observed_fraction':p,
                               'Wilson_CI_low':max(0.,centre-half),'Wilson_CI_high':min(1.,centre+half)})
            bins.extend(points); b=pd.DataFrame(points)
            ax.errorbar(b.Mean_prediction,b.Observed_fraction,
                        yerr=np.maximum(0,np.array([b.Observed_fraction-b.Wilson_CI_low,b.Wilson_CI_high-b.Observed_fraction])),
                        fmt='o-',color=color,capsize=2,markersize=4,lw=1)
            ax.plot([0,1],[0,1],'--',color='#999999',lw=.8)
            m=metrics[(metrics.Model==arm)&(metrics.Dataset==dataset)].iloc[0]
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed event fraction',
                   title=display.get(arm,arm).replace('mViT: ','')+f'\nBrier = {m.Brier:.3f}')
        fig.suptitle(dataset+': descriptive calibration (95% Wilson intervals)')
        fig.tight_layout();save(fig,f'Figure_S4_calibration_{suffix}')
        m=metrics[(metrics.Model.isin(arms))&(metrics.Dataset==dataset)].copy()
        fig,ax=plt.subplots(figsize=(8,2+len(m)*.5))
        for i,row in enumerate(m.itertuples()):
            ax.plot([row.AUC_CI_low,row.AUC_CI_high],[i,i],color=colors[i%len(colors)],lw=2)
            ax.plot(row.AUC,i,'o',color=colors[i%len(colors)])
            ax.text(1.02,i,f'{row.AUC:.3f} [{row.AUC_CI_low:.3f}, {row.AUC_CI_high:.3f}]',
                    transform=ax.get_yaxis_transform(),va='center',fontsize=8)
        ax.set_yticks(range(len(m)),[display.get(x,x).replace('mViT: ','') for x in m.Model])
        ax.invert_yaxis();ax.axvline(.5,color='#999999',ls='--',lw=.8)
        ax.set(xlim=(0,1),xlabel='AUC (95% patient-bootstrap CI)',title=dataset)
        fig.tight_layout();save(fig,f'Figure_S5_AUC_{suffix}')
    if not comparisons.empty:
        fig,ax=plt.subplots(figsize=(10,2+len(comparisons)*.6))
        labels=[]
        for i,row in enumerate(comparisons.itertuples()):
            labels.append(f'{row.Dataset}\n{display.get(row.Model_B,row.Model_B)} − {display.get(row.Model_A,row.Model_A)}')
            ax.plot([row.Delta_CI_low,row.Delta_CI_high],[i,i],color=colors[i%len(colors)],lw=2)
            ax.plot(row.Delta_AUC_B_minus_A,i,'o',color=colors[i%len(colors)])
        ax.set_yticks(range(len(labels)),labels,fontsize=8);ax.invert_yaxis()
        ax.axvline(0,color='#555555',ls='--',lw=.8)
        ax.set(xlabel='Paired AUC difference: Model B − Model A (95% bootstrap CI)',title='Paired model comparisons')
        fig.tight_layout();save(fig,'Figure_S6_paired_differences')
    if not strata.empty and 'AUC' in strata:
        for arm in arms:
            m=strata[strata.Model==arm].copy()
            if m.empty:continue
            fig,ax=plt.subplots(figsize=(9,2+len(m)*.5))
            labels=[]
            for i,row in enumerate(m.itertuples()):
                if row.Variable=='field_strength_t':
                    label=f'Field strength: {row.Level} T'
                else:
                    relation='≤' if str(row.Level).startswith('<=') else '>' if str(row.Level).startswith('>') else None
                    label=f'In-plane spacing {relation} {row.Resolution_cut_mm:.3f} mm' if relation else 'In-plane spacing missing'
                labels.append(f'{label} (n={int(row.N)}, events={int(row.Events)})')
                if np.isfinite(getattr(row,'AUC',np.nan)):
                    ax.plot([row.AUC_CI_low,row.AUC_CI_high],[i,i],color=colors[0],lw=2)
                    ax.plot(row.AUC,i,'o',color=colors[0])
                else:ax.text(.5,i,'Not estimable',ha='center',va='center',fontsize=8)
            ax.set_yticks(range(len(m)),labels,fontsize=8);ax.invert_yaxis()
            ax.set(xlim=(0,1),xlabel='External AUC (95% patient-bootstrap CI)',title=display.get(arm,arm)+' — exploratory subgroups')
            fig.tight_layout();save(fig,f'Figure_S7_subgroups_{arm}')
    return {'Calibration_bins':pd.DataFrame(bins)}


def paired_text(comparisons, display):
    output=[]
    for row in comparisons.itertuples():
        adjusted=getattr(row,'Holm_P',np.nan)
        ptext=f'Holm-adjusted P={adjusted:.3g}' if np.isfinite(adjusted) else f'unadjusted DeLong P={row.DeLong_P:.3g}'
        output.append(f'{row.Dataset}: {display.get(row.Model_B,row.Model_B)} versus {display.get(row.Model_A,row.Model_A)}, '
                      f'paired AUC difference (B minus A) {row.Delta_AUC_B_minus_A:+.3f} '
                      f'(95% CI {row.Delta_CI_low:+.3f} to {row.Delta_CI_high:+.3f}); {ptext}.')
    return output
