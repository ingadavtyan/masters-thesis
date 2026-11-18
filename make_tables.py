"""Builds tables.json (formatted strings, Armenian decimal comma) from results.json and the raw data file."""
import json, sys
import numpy as np, pandas as pd
from scipy import stats

res = json.load(open(sys.argv[1])); data = sys.argv[2]; out = sys.argv[3]

def f(x, d=3):  return f"{x:.{d}f}".replace(".", ",")
def n(x):       return f"{int(x):,}".replace(",", " ")
def pc(x, d=1): return f(100 * x, d)

df = pd.read_csv(data)
a = df["Amount"].values; la = np.log1p(a)
def desc(v):
    return [n(len(v)), f(v.mean(), 2), f(v.std(ddof=1), 2), f(np.median(v), 2), f(np.percentile(v, 25), 2),
            f(np.percentile(v, 75), 2), f(np.percentile(v, 99), 2), f(v.max(), 2), f(stats.skew(v), 2), f(stats.kurtosis(v), 2)]
names = ["Դիտարկումների թիվը", "Միջին", "Ստանդարտ շեղում", "Մեդիան", "Q₁", "Q₃", "99-րդ պերցենտիլ", "Մաքսիմում", "Թեքություն", "Ավելցուկային կուրտոզիս"]
dr, dl = desc(a), desc(la)
t1 = [[names[i], dr[i], dl[i]] for i in range(10)]

A = res["auc"]
def ci(v, k): return f"[{f(v[k][0], 4 if k=='pr_ci' else 3)}; {f(v[k][1], 4 if k=='pr_ci' else 3)}]"
def arow(label, key):
    v = A[key]; return [label, f(v["pr"], 4), ci(v, "pr_ci"), f(v["roc"], 3), ci(v, "roc_ci")]
t2 = [arow("Z (Amount)", "Z (Amount, raw)"), arow("Z / IQR / MAD (log Amount)", "Z (Amount, log)"),
      arow("Z-max", "Z-max"), arow("IQR-max", "IQR-max"), arow("MAD-max", "MAD-max"),
      arow("Mahalanobis", "Mahalanobis"), arow("Mahalanobis-MCD", "Mahalanobis-MCD"),
      arow("Combined-max", "Combined-max"), arow("Combined-mean", "Combined-mean"),
      arow("Isolation Forest", "IsolationForest"), arow("MCD (ռեգուլյարիզացիա չկա)†", "Mahalanobis-MCD (unregularised)")]

def crow(label, thr, v):
    return [label, thr, n(v["tp"]), n(v["fp"]), n(v["fn"]), n(v["tn"]), f(v["precision"], 4), f(v["recall"], 3), f(v["f1"], 4), f(v["fpr"], 4)]
N, S = res["nominal"], res["star"]
t3 = [crow("Z (Amount)", "3,0", N["Z (Amount, raw)"]), crow("Z (log Amount)", "3,0", N["Z (Amount, log)"]),
      crow("IQR (log Amount)", "1,5", N["IQR (Amount, log)"]), crow("MAD (log Amount)", "3,5", N["MAD (Amount, log)"]),
      crow("Z-max*", f(S["Z-max*"]["thr"], 2), S["Z-max*"]), crow("IQR-max*", f(S["IQR-max*"]["thr"], 2), S["IQR-max*"]),
      crow("MAD-max*", f(S["MAD-max*"]["thr"], 2), S["MAD-max*"]),
      crow("Mahalanobis", f(N["Mahalanobis"]["thr"], 2), N["Mahalanobis"]),
      crow("Mahalanobis-MCD", f(N["Mahalanobis-MCD"]["thr"], 2), N["Mahalanobis-MCD"])]

B = res["budget"]["0.99"]
def brow(label, key):
    v = B[key]; return [label, n(v["tp"]), n(v["fp"]), f(v["precision"], 4), f(v["recall"], 3), f(v["f1"], 4), f(v["fpr"], 4)]
t4 = [brow("Z / IQR / MAD (Amount)", "Z (Amount, log)"), brow("Z-max", "Z-max"), brow("IQR-max", "IQR-max"), brow("MAD-max", "MAD-max"),
      brow("Mahalanobis", "Mahalanobis"), brow("Mahalanobis-MCD", "Mahalanobis-MCD"), brow("Combined-max", "Combined-max"),
      brow("Isolation Forest", "IsolationForest")]

facts = dict(
    base_rate=f(res["test_fraud"] / res["test_n"], 4),
    alert_test={k: pc((res["nominal"][k]["tp"] + res["nominal"][k]["fp"]) / res["test_n"]) for k in ("Z-max", "IQR-max", "MAD-max")},
    alert_train={k: pc(res["nominal"][k]["train_alert_rate"]) for k in res["nominal"]},
    star_train={k: pc(v["train_alert_rate"]) for k, v in res["star"].items()},
    cond_raw=f"{res['mcd_raw_eigmax'] / res['mcd_raw_eigmin']:.2e}",
    cond_cls=f"{res['cls_eigmax'] / res['cls_eigmin']:.1f}",
    floor={k: [f(v["pr"], 3), f(v["roc"], 3)] for k, v in res["floor_sens"].items()},
    dom_fraud=res["dominant_feat_fraud"], dom_legit={k: pc(v / (res["test_n"] - res["test_fraud"])) for k, v in res["dominant_feat_legit"].items()},
)
json.dump(dict(t1=t1, t2=t2, t3=t3, t4=t4, facts=facts), open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(facts, ensure_ascii=False, indent=1))
