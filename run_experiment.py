"""
Ավանդական և մեքենայական ուսուցման մոդելների համեմատական փորձ
բանկային խարդախ գործարքների հայտնաբերման խնդրում։

Run:  python src/run_experiment.py
"""
import json, time, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, roc_curve, precision_recall_curve,
                             confusion_matrix)
from xgboost import XGBClassifier

from data_generator import generate

OUT = "results"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True); os.makedirs("data", exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 150})
REVIEW_COST = 5.0          # մեկ կեղծ ահազանգի ստուգման ծախսը (USD)

# ---------------- 1. Տվյալներ ----------------
df = generate()
df.to_csv("data/synthetic_transactions.csv", index=False)
NUM = ["amount", "amount_to_avg_ratio", "hour", "tx_count_1h", "distance_from_home_km",
       "is_foreign", "is_new_device", "card_present", "failed_attempts_24h",
       "customer_age", "account_age_days"]
CAT = ["merchant_category"]
df["log_amount"] = np.log1p(df.amount)
NUM = ["log_amount" if c == "amount" else c for c in NUM]

# Ժամանակային բաժանում (out-of-time)՝ 0-119 օր ուսուցում, 120-139 վալիդացիա, 140-179 թեստ
tr, va, te = df[df.day < 120], df[(df.day >= 120) & (df.day < 140)], df[df.day >= 140]
X_tr, y_tr = tr[NUM + CAT], tr.is_fraud.values
X_va, y_va = va[NUM + CAT], va.is_fraud.values
X_te, y_te = te[NUM + CAT], te.is_fraud.values
stats = {"n_total": len(df), "fraud_rate": float(df.is_fraud.mean()),
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "fraud_train": int(y_tr.sum()), "fraud_val": int(y_va.sum()), "fraud_test": int(y_te.sum())}

prep_scaled = ColumnTransformer([("num", StandardScaler(), NUM),
                                 ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
prep_raw = ColumnTransformer([("num", "passthrough", NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

# ---------------- 2. Կանոնային (rule-based) համակարգ ----------------
def rule_score(X):
    r1 = (X.amount_to_avg_ratio > 4) & (X.is_foreign == 1)
    r2 = X.tx_count_1h >= 4
    r3 = (X.is_new_device == 1) & (X.amount_to_avg_ratio > 3)
    r4 = X.failed_attempts_24h >= 3
    r5 = X.merchant_category.isin(["money_transfer", "gambling", "jewelry"]) & (X.amount_to_avg_ratio > 3)
    r6 = (X.hour <= 5) & (X.card_present == 0) & (X.amount_to_avg_ratio > 2)
    return (r1.astype(int) + r2 + r3 + r4 + r5 + r6).values.astype(float)

# ---------------- 3. Մոդելներ ----------------
models = {
    "Logistic regression": Pipeline([("p", prep_scaled),
        ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Decision tree": Pipeline([("p", prep_raw),
        ("m", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=0))]),
    "Random forest": Pipeline([("p", prep_raw),
        ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=0))]),
    "XGBoost": Pipeline([("p", prep_raw),
        ("m", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
                            n_jobs=-1, random_state=0))]),
    "Neural network (MLP)": Pipeline([("p", prep_scaled),
        ("m", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=200,
                            early_stopping=True, random_state=0))]),
}

scores_va, scores_te, times = {}, {}, {}
scores_va["Rule-based system"] = rule_score(X_va); scores_te["Rule-based system"] = rule_score(X_te)
times["Rule-based system"] = 0.0

for name, m in models.items():
    t = time.time(); m.fit(X_tr, y_tr); times[name] = time.time() - t
    scores_va[name] = m.predict_proba(X_va)[:, 1]; scores_te[name] = m.predict_proba(X_te)[:, 1]
    print(f"{name}: {times[name]:.1f}s")

# Isolation Forest (առանց վերահսկման՝ չի օգտագործում պիտակները)
iso = Pipeline([("p", prep_scaled), ("m", IsolationForest(n_estimators=300, contamination="auto", random_state=0))])
t = time.time(); iso.fit(X_tr); times["Isolation forest"] = time.time() - t
scores_va["Isolation forest"] = -iso.score_samples(X_va); scores_te["Isolation forest"] = -iso.score_samples(X_te)

# Հիբրիդ՝ կանոնային ազդանշաններ + XGBoost գնահատական (ինտեգրման օրինակ)
hyb_tr = np.column_stack([rule_score(X_tr), models["XGBoost"].predict_proba(X_tr)[:, 1]])
hybrid = LogisticRegression(max_iter=1000, class_weight="balanced")
# Հիբրիդի մետա-մոդելը ուսուցանվում է վալիդացիոն բազմության վրա՝ գերուսուցումից խուսափելու համար
hyb_va = np.column_stack([scores_va["Rule-based system"], scores_va["XGBoost"]])
hyb_te = np.column_stack([scores_te["Rule-based system"], scores_te["XGBoost"]])
hybrid.fit(hyb_va, y_va)
scores_va["Hybrid (rules + XGBoost)"] = hybrid.predict_proba(hyb_va)[:, 1]
scores_te["Hybrid (rules + XGBoost)"] = hybrid.predict_proba(hyb_te)[:, 1]
times["Hybrid (rules + XGBoost)"] = times["XGBoost"]

# ---------------- 4. Գնահատում ----------------
def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = np.nanargmax(f1[:-1]); return t[i]

amt_te = te.amount.values
rows = []
for name in scores_te:
    s_va, s_te = scores_va[name], scores_te[name]
    thr = best_threshold(y_va, s_va)             # շեմը ընտրվում է վալիդացիայի վրա
    pred = (s_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    # Ֆինանսական արդյունք․ բաց թողնված խարդախության կորուստ + ստուգման ծախսեր
    loss_fn = amt_te[(y_te == 1) & (pred == 0)].sum()
    cost = loss_fn + REVIEW_COST * (tp + fp)
    total_fraud = amt_te[y_te == 1].sum()
    # Recall, երբ ահազանգվում է գործարքների ընդամենը 1%-ը
    k = int(0.01 * len(s_te)); top = np.argsort(-s_te)[:k]
    rows.append({"model": name,
                 "ROC_AUC": roc_auc_score(y_te, s_te), "PR_AUC": average_precision_score(y_te, s_te),
                 "Precision": precision_score(y_te, pred, zero_division=0),
                 "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred),
                 "Recall@1%": y_te[top].sum() / y_te.sum(),
                 "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
                 "Total_cost_USD": cost, "Saved_share": 1 - cost / total_fraud,
                 "Train_time_s": times[name]})
res = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
res.to_csv(f"{OUT}/metrics.csv", index=False)
stats["total_fraud_amount_test"] = float(amt_te[y_te == 1].sum())
json.dump(stats, open(f"{OUT}/data_stats.json", "w"), indent=2)
print(res.round(4).to_string(index=False)); print(stats)

# Լոգիստիկ ռեգրեսիայի գործակիցներ (մեկնաբանելիություն)
lr = models["Logistic regression"]
names = list(NUM) + list(lr.named_steps["p"].named_transformers_["cat"].get_feature_names_out(CAT))
coef = pd.DataFrame({"feature": names, "coef": lr.named_steps["m"].coef_[0]})
coef["odds_ratio"] = np.exp(coef.coef); coef.sort_values("coef", key=abs, ascending=False).to_csv(f"{OUT}/logit_coefficients.csv", index=False)

# XGBoost հատկանիշների կարևորություն (gain)
xg = models["XGBoost"]
names_raw = list(NUM) + list(xg.named_steps["p"].named_transformers_["cat"].get_feature_names_out(CAT))
imp = pd.Series(xg.named_steps["m"].feature_importances_, index=names_raw).sort_values(ascending=False)
imp.to_csv(f"{OUT}/xgb_importance.csv")

# ---------------- 5. Գրաֆիկներ ----------------
LBL = {"Rule-based system": "Կանոնային համակարգ", "Logistic regression": "Լոգիստիկ ռեգրեսիա",
       "Decision tree": "Որոշումների ծառ", "Random forest": "Պատահական անտառ", "XGBoost": "XGBoost",
       "Neural network (MLP)": "Նեյրոնային ցանց (MLP)", "Isolation forest": "Isolation Forest",
       "Hybrid (rules + XGBoost)": "Հիբրիդ (կանոններ + XGBoost)"}
order = [m for m in LBL if m != "Hybrid (rules + XGBoost)"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for n in order:
    fpr, tpr, _ = roc_curve(y_te, scores_te[n]); ax[0].plot(fpr, tpr, lw=1.5, label=LBL[n])
    p, r, _ = precision_recall_curve(y_te, scores_te[n]); ax[1].plot(r, p, lw=1.5, label=LBL[n])
ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].set_xlabel("Կեղծ դրական հաճախականություն (FPR)")
ax[0].set_ylabel("Ճշմարիտ դրական հաճախականություն (TPR)"); ax[0].set_title("ա) ROC կորեր")
ax[1].set_xlabel("Ամբողջականություն (Recall)"); ax[1].set_ylabel("Ճշգրտություն (Precision)")
ax[1].set_title("բ) Precision–Recall կորեր"); ax[1].legend(fontsize=7.5, loc="upper right")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/roc_pr_curves.png"); plt.close()

r2 = res.set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(9, 4.2)); x = np.arange(len(order)); w = .27
ax.bar(x - w, r2.PR_AUC, w, label="PR-AUC", color="#2b5c8a")
ax.bar(x, r2.F1, w, label="F1", color="#e08a3c")
ax.bar(x + w, r2.Recall, w, label="Recall", color="#6aa56a")
ax.set_xticks(x); ax.set_xticklabels([LBL[n] for n in order], rotation=25, ha="right", fontsize=8)
ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3); ax.set_title("Մոդելների որակի հիմնական չափանիշները (թեստային բազմություն)")
plt.tight_layout(); plt.savefig(f"{FIG}/metrics_bar.png"); plt.close()

top = imp.head(10)[::-1]
HY = {"log_amount": "Գումար (log)", "amount_to_avg_ratio": "Գումար / միջին", "hour": "Ժամ",
      "tx_count_1h": "Գործարքներ 1 ժամում", "distance_from_home_km": "Հեռավորություն (կմ)",
      "is_foreign": "Արտասահմանյան", "is_new_device": "Նոր սարք", "card_present": "Քարտը ներկա է",
      "failed_attempts_24h": "Անհաջող փորձեր 24ժ", "customer_age": "Հաճախորդի տարիք",
      "account_age_days": "Հաշվի տարիք (օր)"}
lab = [HY.get(i, i.replace("merchant_category_", "Կատեգորիա: ")) for i in top.index]
fig, ax = plt.subplots(figsize=(7.5, 4)); ax.barh(lab, top.values, color="#2b5c8a")
ax.set_xlabel("Կարևորություն (gain)"); ax.set_title("XGBoost մոդելի 10 առավել կարևոր հատկանիշները"); ax.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/xgb_importance.png"); plt.close()

# Դասերի բաշխում և գումարների բաշխում
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].hist(np.log1p(df.amount[df.is_fraud == 0]), bins=60, alpha=.6, density=True, label="Օրինական")
ax[0].hist(np.log1p(df.amount[df.is_fraud == 1]), bins=60, alpha=.6, density=True, label="Խարդախ")
ax[0].set_xlabel("ln(1 + գումար)"); ax[0].set_title("ա) Գործարքի գումարի բաշխումը"); ax[0].legend()
hr = df.groupby("hour").is_fraud.mean() * 100
ax[1].bar(hr.index, hr.values, color="#c0504d"); ax[1].set_xlabel("Օրվա ժամ"); ax[1].set_ylabel("Խարդախության մասնաբաժին, %")
ax[1].set_title("բ) Խարդախության մակարդակն ըստ ժամի")
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/data_overview.png"); plt.close()
print("done")
