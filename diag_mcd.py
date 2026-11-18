import numpy as np, pandas as pd, sys
from sklearn.covariance import MinCovDet
from sklearn.metrics import average_precision_score, roc_auc_score
df = pd.read_csv("/home/claude/data/creditcard.csv").sort_values("Time", kind="stable").reset_index(drop=True)
cut = int(0.7*len(df)); tr, te = df.iloc[:cut], df.iloc[cut:]
feats=[f"V{i}" for i in range(1,29)]
def prep(d):
    X=d[feats].copy(); X["logAmt"]=np.log1p(d["Amount"]); return X.values
Xtr, Xte = prep(tr), prep(te); yte = te.Class.values
S = np.cov(Xtr, rowvar=False); ev = np.linalg.eigvalsh(S)
print("classical cov eig min/max/cond", ev.min(), ev.max(), ev.max()/ev.min())
def md2(X,m,Si): D=X-m; return np.einsum("ij,jk,ik->i",D,Si,D)
rng=np.random.default_rng(0)
for sf in (0.75, 0.9):
    for seed in (1,2,3):
        idx=np.random.default_rng(seed).choice(len(Xtr),60000,replace=False)
        m=MinCovDet(support_fraction=sf, random_state=seed).fit(Xtr[idx])
        e=np.linalg.eigvalsh(m.covariance_)
        s=md2(Xte,m.location_,np.linalg.inv(m.covariance_))
        st=md2(Xtr,m.location_,np.linalg.inv(m.covariance_))
        print(f"sf={sf} seed={seed} eigmin={e.min():.3e} eigmax={e.max():.3e} cond={e.max()/e.min():.2e} med_train_score={np.median(st):.1f} PR={average_precision_score(yte,s):.4f} ROC={roc_auc_score(yte,s):.4f}")
# which direction is degenerate (seed 1 sf 0.9)
idx=np.random.default_rng(1).choice(len(Xtr),60000,replace=False)
m=MinCovDet(support_fraction=0.9, random_state=1).fit(Xtr[idx])
w,V=np.linalg.eigh(m.covariance_); v=V[:,0]
print("smallest-eig direction top loadings:", [(( feats+['logAmt'])[i], round(v[i],3)) for i in np.argsort(-np.abs(v))[:5]])
print("raw train var of those:", Xtr.var(0)[np.argsort(-np.abs(v))[:5]])
print("support size", m.support_.sum(), "dist from support: logAmt zero share in support", (Xtr[idx][m.support_][:,-1]==0).mean(), "overall", (Xtr[:,-1]==0).mean())
