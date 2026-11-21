"""Patient-level discrimination, operating metrics, calibration and paired tests."""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_curve


def auc(y,p):
    y=np.asarray(y); p=np.asarray(p)
    n1=int(y.sum());n0=len(y)-n1
    if min(n1,n0)==0:return np.nan
    return float((rankdata(p)[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def threshold(y,p):
    fpr,tpr,cut=roc_curve(y,p,drop_intermediate=False)
    valid=np.flatnonzero((tpr>=.95)&np.isfinite(cut))
    best=valid[np.isclose(fpr[valid],fpr[valid].min())]
    return float(cut[best].min())


def calibration(y,p):
    """Joint intercept and slope, not offset-only calibration-in-the-large."""
    x=np.c_[np.ones(len(y)),logit(np.clip(p,1e-6,1-1e-6))]
    beta=np.array([0.,1.])
    for _ in range(50):
        eta=x@beta;q=expit(eta)
        h=x.T@((q*(1-q))[:,None]*x)
        g=x.T@(q-y)
        try:step=np.linalg.solve(h,g)
        except np.linalg.LinAlgError:return np.array([np.nan,np.nan])
        loss=np.sum(np.logaddexp(0,eta)-y*eta)
        scale=1.
        while scale>1e-6:
            proposal=beta-scale*step;new=x@proposal
            if np.sum(np.logaddexp(0,new)-y*new)<=loss+1e-9:break
            scale*=.5
        beta=proposal
        if np.max(np.abs(beta))>100:return np.array([np.nan,np.nan])
        if np.max(np.abs(scale*step))<1e-7:return beta
    return np.array([np.nan,np.nan])


def estimates(y,p,cut):
    y=np.asarray(y,dtype=int);p=np.asarray(p,float)
    intercept,slope=calibration(y,p)
    output={'AUC':auc(y,p),'Brier':float(np.mean((p-y)**2)),
            'Calibration_intercept_joint':intercept,'Calibration_slope_joint':slope}
    if cut is not None:
        positive=p>=cut
        tp=int(np.sum(positive&(y==1)));fp=int(np.sum(positive&(y==0)))
        tn=int(np.sum(~positive&(y==0)));fn=int(np.sum(~positive&(y==1)))
        div=lambda a,b:a/b if b else np.nan
        output.update(Sensitivity=div(tp,tp+fn),Specificity=div(tn,tn+fp),PPV=div(tp,tp+fp),
                      NPV=div(tn,tn+fn),Accuracy=div(tp+tn,len(y)),TP=tp,FP=fp,TN=tn,FN=fn)
    return output


def metrics_ci(y,p,cut=None,rounds=2000,seed=42):
    y=np.asarray(y,int);p=np.asarray(p,float)
    out={'N':len(y),'Events':int(y.sum()),'Threshold':cut,**estimates(y,p,cut)}
    if min(y.sum(),len(y)-y.sum())<2:
        out['CI_status']='insufficient_classes';return out
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    names=[k for k in out if k not in ['N','Events','Threshold','TP','FP','TN','FN']]
    replicates=[]
    for _ in range(rounds):
        ix=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        v=estimates(y[ix],p[ix],cut)
        replicates.append([v[k] for k in names])
    boot=np.asarray(replicates)
    for j,name in enumerate(names):
        valid=boot[:,j][np.isfinite(boot[:,j])]
        out[name+'_valid_bootstraps']=len(valid)
        if len(valid)>=.9*rounds:
            out[name+'_CI_low'],out[name+'_CI_high']=np.quantile(valid,[.025,.975])
        else:out[name+'_CI_low']=out[name+'_CI_high']=np.nan
    out['Bootstrap_replicates']=rounds
    return out


def paired(y,a,b,rounds=2000,seed=42):
    """Difference B minus A. DeLong covariance uses paired placement values."""
    y=np.asarray(y,int);a=np.asarray(a,float);b=np.asarray(b,float)
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    v10=[];v01=[]
    for p in [a,b]:
        greater=p[pos,None]>p[None,neg]
        ties=p[pos,None]==p[None,neg]
        comparison=greater+.5*ties
        v10.append(comparison.mean(1));v01.append(comparison.mean(0))
    cov=np.cov(v10)/len(pos)+np.cov(v01)/len(neg)
    difference=auc(y,b)-auc(y,a)
    variance=cov[0,0]+cov[1,1]-2*cov[0,1]
    pvalue=2*norm.sf(abs(difference)/np.sqrt(variance)) if variance>1e-15 else (1. if abs(difference)<1e-12 else 0.)
    rng=np.random.default_rng(seed);diff=[]
    for _ in range(rounds):
        ix=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
        diff.append(auc(y[ix],b[ix])-auc(y[ix],a[ix]))
    low,high=np.quantile(diff,[.025,.975])
    return {'Delta_AUC_B_minus_A':difference,'Delta_CI_low':low,'Delta_CI_high':high,
            'DeLong_P':float(pvalue),'Bootstrap_replicates':rounds}


def holm(values):
    values=np.asarray(values,float);order=np.argsort(values)
    adjusted=np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result=np.empty_like(adjusted);result[order]=adjusted
    return result
