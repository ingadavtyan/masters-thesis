"""Reproducible experiment: statistical anomaly detection on the ULB Credit Card Fraud dataset.
Usage: python experiment.py creditcard.csv outdir
"""
import sys, json, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out = sys.argv[1], sys.argv[2]
SEED = 42
rng = np.random.default_rng(SEED)
df = pd.read_csv(path).sort_values("Time", kind="stable").reset_index(drop=True)
res = {}
res["n"], res["n_fraud"] = int(len(df)), int(df.Class.sum())
res["missing"] = int(df.isna().sum().sum()); res["duplicates"] = int(df.duplicated().sum())

# ---------- descriptive statistics ----------
a = df["Amount"]
res["amount_desc"] = dict(mean=a.mean(), sd=a.std(), median=a.median(), q1=a.quantile(.25), q3=a.quantile(.75),
    p99=a.quantile(.99), max=a.max(), skew=stats.skew(a), kurt=stats.kurtosis(a),
    skew_log=stats.skew(np.log1p(a)), kurt_log=stats.kurtosis(np.log1p(a)), zero=int((a==0).sum()))
res["amount_by_class"] = {int(c): dict(mean=g.mean(), median=g.median(), sd=g.std()) for c, g in a.groupby(df.Class)}

# ---------- chronological split ----------
cut = int(0.7 * len(df))
tr, te = df.iloc[:cut].copy(), df.iloc[cut:].copy()
res["train_n"], res["train_fraud"] = int(len(tr)), int(tr.Class.sum())
res["test_n"], res["test_fraud"] = int(len(te)), int(te.Class.sum())
res["cut_time"] = float(tr.Time.iloc[-1])

feats = [f"V{i}" for i in range(1, 29)]
def prep(d):                       # features: V1..V28 and log(1+Amount); Time is excluded
    X = d[feats].copy(); X["logAmt"] = np.log1p(d["Amount"]); return X.values
Xtr, Xte = prep(tr), prep(te)
ytr, yte = tr.Class.values, te.Class.values
p = Xtr.shape[1]
amt_mu, amt_sd = tr["Amount"].mean(), tr["Amount"].std(ddof=1)

# ---------- parameters estimated on TRAIN only (labels not used) ----------
mu, sd = Xtr.mean(0), Xtr.std(0, ddof=1)
med = np.median(Xtr, 0); mad = np.median(np.abs(Xtr - med), 0)
q1, q3 = np.percentile(Xtr, 25, 0), np.percentile(Xtr, 75, 0); iqr = q3 - q1
Sigma = np.cov(Xtr, rowvar=False); Sinv = np.linalg.inv(Sigma)
C = np.corrcoef(Xtr, rowvar=False)
res["max_abs_corr_offdiag"] = float(np.max(np.abs(C - np.eye(p))))
res["maxcorr_logAmt"] = float(np.max(np.abs(C[-1, :-1])))
t0 = time.time()
idx = rng.choice(len(Xtr), 60000, replace=False)
mcd = MinCovDet(support_fraction=0.9, random_state=SEED).fit(Xtr[idx])
res["mcd_seconds"] = time.time() - t0
w_raw, V_raw = np.linalg.eigh(mcd.covariance_)
res["mcd_raw_eigmin"], res["mcd_raw_eigmax"] = float(w_raw.min()), float(w_raw.max())
res["cls_eigmin"], res["cls_eigmax"] = float(np.linalg.eigvalsh(Sigma).min()), float(np.linalg.eigvalsh(Sigma).max())
def reg_inv(floor):
    return (V_raw / np.maximum(w_raw, floor)) @ V_raw.T
FLOOR = res["cls_eigmin"]            # label-free rule: robust variance in any direction >= smallest classical eigenvalue
Sinv_r = reg_inv(FLOOR)
Sinv_raw = np.linalg.inv(mcd.covariance_)

# ---------- anomaly scores (larger = more anomalous) ----------
def z_amount_raw(d): return np.abs((d["Amount"].values - amt_mu) / amt_sd)
def z_amount_log(d): return np.abs((np.log1p(d["Amount"].values) - mu[-1]) / sd[-1])
def iqr_amount_log(d):
    x = np.log1p(d["Amount"].values); return np.maximum((q1[-1]-x)/iqr[-1], (x-q3[-1])/iqr[-1])
def mad_amount_log(d): return np.abs(0.6745*(np.log1p(d["Amount"].values)-med[-1])/mad[-1])
def z_max(X): return np.max(np.abs((X-mu)/sd), 1)
def iqr_max(X): return np.max(np.maximum((q1-X)/iqr, (X-q3)/iqr), 1)
def mad_max(X): return np.max(np.abs(0.6745*(X-med)/mad), 1)
def md2(X, m, Si): D = X-m; return np.einsum("ij,jk,ik->i", D, Si, D)
def md_cl(X): return md2(X, mu, Sinv)
def md_rb(X): return md2(X, mcd.location_, Sinv_r)
def md_rb_raw(X): return md2(X, mcd.location_, Sinv_raw)

scores_tr, scores_te = {}, {}
S = {"Z (Amount, raw)": (z_amount_raw, "df"), "Z (Amount, log)": (z_amount_log, "df"),
     "IQR (Amount, log)": (iqr_amount_log, "df"), "MAD (Amount, log)": (mad_amount_log, "df"),
     "Z-max": (z_max, "X"), "IQR-max": (iqr_max, "X"), "MAD-max": (mad_max, "X"),
     "Mahalanobis": (md_cl, "X"), "Mahalanobis-MCD": (md_rb, "X"), "Mahalanobis-MCD (unregularised)": (md_rb_raw, "X")}
for k, (f, kind) in S.items():
    scores_tr[k] = f(tr if kind == "df" else Xtr); scores_te[k] = f(te if kind == "df" else Xte)
# combined: empirical-CDF normalisation on TRAIN scores
def ecdf(ref):
    r = np.sort(ref); return lambda s: np.searchsorted(r, s, side="right") / len(r)
Fm, Fd = ecdf(scores_tr["MAD-max"]), ecdf(scores_tr["Mahalanobis-MCD"])
for sc in (scores_tr, scores_te):
    um, ud = Fm(sc["MAD-max"]), Fd(sc["Mahalanobis-MCD"])
    sc["Combined-max"] = np.maximum(um, ud); sc["Combined-mean"] = 0.5*(um+ud)
iso = IsolationForest(n_estimators=200, max_samples=256, random_state=SEED, n_jobs=-1).fit(Xtr)
scores_tr["IsolationForest"] = -iso.score_samples(Xtr); scores_te["IsolationForest"] = -iso.score_samples(Xte)

# ---------- evaluation ----------
def conf(y, flag):
    tp = int(((flag==1)&(y==1)).sum()); fp = int(((flag==1)&(y==0)).sum())
    fn = int(((flag==0)&(y==1)).sum()); tn = int(((flag==0)&(y==0)).sum())
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn); F = 2*P*R/(P+R) if P+R else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=P, recall=R, f1=F, fpr=fp/(fp+tn), alerts=tp+fp)

nominal = {"Z (Amount, raw)": 3.0, "Z (Amount, log)": 3.0, "IQR (Amount, log)": 1.5, "MAD (Amount, log)": 3.5,
           "Z-max": 3.0, "IQR-max": 1.5, "MAD-max": 3.5,
           "Mahalanobis": stats.chi2.ppf(0.999, p), "Mahalanobis-MCD": stats.chi2.ppf(0.999, p)}
res["chi2_thr"] = float(stats.chi2.ppf(0.999, p))
res["auc"] = {k: dict(pr=average_precision_score(yte, scores_te[k]), roc=roc_auc_score(yte, scores_te[k])) for k in scores_te}
res["floor_sens"] = {}
for lab, fl in (("1e-4*max", 1e-4*w_raw.max()), ("1e-3*max", 1e-3*w_raw.max()), ("1e-2*max", 1e-2*w_raw.max()),
                ("classical min eig", FLOOR), ("0.5", 0.5)):
    st = md2(Xte, mcd.location_, reg_inv(fl))
    res["floor_sens"][lab] = dict(floor=float(fl), pr=average_precision_score(yte, st), roc=roc_auc_score(yte, st))
# multiplicity-corrected (Bonferroni, alpha=0.001 over p features) Gaussian-equivalent thresholds
zstar = float(stats.norm.ppf(1 - 0.001/(2*p))); res["zstar"] = zstar
star = {"Z-max*": ("Z-max", zstar), "IQR-max*": ("IQR-max", (zstar-0.6745)/1.349), "MAD-max*": ("MAD-max", zstar)}
res["star"] = {lab: dict(thr=float(t), **conf(yte, (scores_te[k] > t).astype(int)),
                         train_alert_rate=float((scores_tr[k] > t).mean())) for lab, (k, t) in star.items()}
res["nominal"] = {k: dict(thr=float(t), **conf(yte, (scores_te[k] > t).astype(int)),
                          train_alert_rate=float((scores_tr[k] > t).mean())) for k, t in nominal.items()}
# alert-budget thresholds: TRAIN quantile of the score (no test information)
res["budget"] = {}
for q in (0.99, 0.995, 0.999):
    res["budget"][str(q)] = {}
    for k in scores_te:
        t = np.quantile(scores_tr[k], q)
        res["budget"][str(q)][k] = dict(thr=float(t), **conf(yte, (scores_te[k] > t).astype(int)))
grids = {"Z-max": [2.5, 3, 3.5, 4, 5, 6], "IQR-max": [1.5, 2, 3, 4, 6, 8], "MAD-max": [3.5, 4, 5, 6, 8, 10]}
res["sens"] = {k: {str(g): conf(yte, (scores_te[k] > g).astype(int)) for g in v} for k, v in grids.items()}
for k in ("Mahalanobis-MCD", "Mahalanobis"):
    res["sens"][k] = {str(al): conf(yte, (scores_te[k] > stats.chi2.ppf(al, p)).astype(int))
                      for al in (0.95, 0.99, 0.999, 0.9999, 0.99999)}
# bootstrap CIs (stratified)
B = 300; pos, neg = np.where(yte==1)[0], np.where(yte==0)[0]
boot = {k: [] for k in scores_te}
for b in range(B):
    ii = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for k in scores_te:
        boot[k].append((average_precision_score(yte[ii], scores_te[k][ii]), roc_auc_score(yte[ii], scores_te[k][ii])))
for k in boot:
    arr = np.array(boot[k]); res["auc"][k]["pr_ci"] = np.percentile(arr[:,0],[2.5,97.5]).tolist()
    res["auc"][k]["roc_ci"] = np.percentile(arr[:,1],[2.5,97.5]).tolist()
res["frauds_in_test_flagged_by_both"] = int(((scores_te["MAD-max"]>3.5)&(scores_te["Mahalanobis-MCD"]>nominal["Mahalanobis-MCD"])&(yte==1)).sum())
Zm = np.abs(0.6745*(Xte-med)/mad)
names = np.array(feats+["logAmt"])
res["dominant_feat_fraud"] = pd.Series(names[Zm[yte==1].argmax(1)]).value_counts().head(5).to_dict()
res["dominant_feat_legit"] = pd.Series(names[Zm[yte==0].argmax(1)]).value_counts().head(5).to_dict()
res["test_fraud_amt_median"] = float(te.Amount[yte==1].median()); res["test_legit_amt_median"] = float(te.Amount[yte==0].median())

# ---------- figures ----------
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(1, 2, figsize=(5.1, 1.6))
ax[0].hist(df.Amount[df.Amount < 500], bins=60, color="#4c72b0"); ax[0].set_yscale("log")
ax[0].set_xlabel("Amount (<500)"); ax[0].set_ylabel("count (log)"); ax[0].set_title("(a)", fontsize=7)
ax[1].hist(np.log1p(df.Amount), bins=60, color="#55a868"); ax[1].set_xlabel("log(1+Amount)"); ax[1].set_title("(b)", fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}/fig1_amount.png", dpi=300); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(5.1, 1.7))
for a_, k in zip(ax, ("MAD-max", "Mahalanobis", "Mahalanobis-MCD")):
    s = np.log10(scores_te[k]+1e-9)
    bins = np.linspace(np.percentile(s, .1), np.percentile(s, 99.99), 60)
    a_.hist(s[yte==0], bins=bins, density=True, alpha=.6, label="legit", color="#4c72b0")
    a_.hist(s[yte==1], bins=bins, density=True, alpha=.6, label="fraud", color="#c44e52")
    a_.set_title(k, fontsize=7); a_.set_xlabel("log10(score)")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{out}/fig2_scores.png", dpi=300); plt.close()

cols = {"Z-max": "#1f77b4", "IQR-max": "#ff7f0e", "MAD-max": "#2ca02c", "Mahalanobis": "#d62728",
        "Mahalanobis-MCD": "#9467bd", "Combined-max": "#8c564b", "IsolationForest": "#e377c2"}
fig, ax = plt.subplots(1, 2, figsize=(5.1, 2.05))
for k in cols:
    P, R, _ = precision_recall_curve(yte, scores_te[k]); ax[0].plot(R, P, color=cols[k], lw=1, label=k)
ax[0].axhline(yte.mean(), color="k", ls="--", lw=.8)
ax[0].set_yscale("log"); ax[0].set_ylim(5e-4, 1.5); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision (log scale)")
ax[0].set_title("(a)", fontsize=7); ax[0].legend(fontsize=4.5, loc="upper right")
for k in ("Z-max", "IQR-max", "MAD-max", "Mahalanobis", "Mahalanobis-MCD"):
    pts = res["sens"][k]
    ax[1].plot([v["fpr"] for v in pts.values()], [v["recall"] for v in pts.values()], marker="o", ms=2.5, lw=1, color=cols[k], label=k)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("Recall"); ax[1].set_title("(b)", fontsize=7)
ax[1].legend(fontsize=5, loc="lower right")
plt.tight_layout(); plt.savefig(f"{out}/fig3_pr_recall.png", dpi=300); plt.close()

json.dump(res, open(f"{out}/results.json", "w"), indent=1, default=float)
print("done")
