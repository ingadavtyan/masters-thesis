"""
Սինթետիկ բանկային գործարքների տվյալների գեներատոր։
Synthetic bank card transaction generator for fraud-detection experiments.

Տվյալները իրական չեն․ դրանք մոդելավորում են իրական քարտային գործարքների
հիմնական վիճակագրական հատկությունները (դասերի խիստ անհավասարակշռվածություն,
ոչ գծային փոխազդեցություններ, աղմուկ, ժամանակային կառուցվածք)։
"""
import numpy as np
import pandas as pd

MERCHANTS = {  # կատեգորիա: (հաճախականություն, ռիսկի բազային մակարդակ)
    "grocery": (0.30, 0.2), "restaurant": (0.15, 0.3), "transport": (0.10, 0.3),
    "utilities": (0.08, 0.1), "clothing": (0.10, 0.5), "electronics": (0.07, 1.2),
    "travel": (0.05, 1.0), "online_services": (0.08, 1.1), "jewelry": (0.02, 1.6),
    "gambling": (0.02, 1.8), "money_transfer": (0.03, 1.9),
}


def generate(n=300_000, n_customers=8_000, target_fraud_rate=0.006, seed=42):
    rng = np.random.default_rng(seed)

    # --- Հաճախորդների պրոֆիլներ ---
    cust_avg = rng.lognormal(mean=3.4, sigma=0.7, size=n_customers)      # միջին գործարք (USD)
    cust_age = rng.integers(18, 80, size=n_customers)
    cust_acc_age = rng.integers(10, 4000, size=n_customers)              # հաշվի տարիք (օր)

    cid = rng.integers(0, n_customers, size=n)
    day = np.sort(rng.integers(0, 180, size=n))                          # 6 ամիս
    hour = rng.choice(24, size=n, p=_hour_profile())

    cats = list(MERCHANTS)
    p_cat = np.array([MERCHANTS[c][0] for c in cats]); p_cat /= p_cat.sum()
    cat = rng.choice(len(cats), size=n, p=p_cat)
    cat_risk = np.array([MERCHANTS[c][1] for c in cats])[cat]

    amount = cust_avg[cid] * rng.lognormal(0, 0.6, size=n)
    ratio = amount / cust_avg[cid]
    tx_1h = rng.poisson(0.4, size=n)
    distance = rng.exponential(8, size=n)
    foreign = rng.random(n) < 0.04
    new_device = rng.random(n) < 0.05
    card_present = rng.random(n) < np.where(np.isin(cat, [cats.index("online_services"),
                                                         cats.index("travel")]), 0.1, 0.75)
    failed_24h = rng.poisson(0.08, size=n)
    night = ((hour >= 0) & (hour <= 5)).astype(float)

    # --- Խարդախության թաքնված (latent) ռիսկ՝ ոչ գծային փոխազդեցություններով ---
    z = (-1.0
         + 0.35 * np.log1p(ratio)
         + 0.30 * cat_risk
         + 0.20 * tx_1h
         + 0.30 * foreign + 0.30 * new_device - 0.30 * card_present
         + 0.20 * failed_24h
         # փոխազդեցություններ (գծային մոդելը դրանք ուղղակի չի տեսնում)
         + 2.2 * night * foreign
         + 2.0 * new_device * (ratio > 3)
         + 1.8 * (tx_1h >= 3) * (~card_present)
         + 1.8 * (distance > 40) * (cust_acc_age[cid] < 365)
         + 1.5 * (cat_risk > 1.5) * (ratio > 2.5)
         # ոչ մոնոտոն ազդեցություններ՝ տարիք և օրվա ժամ
         + 0.45 * ((cust_age[cid] - 45) / 15) ** 2
         + 0.6 * np.cos((hour - 3) / 24 * 2 * np.pi)
         + rng.normal(0, 0.9, size=n))                                    # աղմուկ

    # Կալիբրացիա՝ ցանկալի խարդախության մակարդակին հասնելու համար
    thr = np.quantile(z, 1 - target_fraud_rate * 0.85)
    fraud = (z > thr)
    fraud |= rng.random(n) < target_fraud_rate * 0.15                    # «պատահական» խարդախներ

    # Խարդախ գործարքները փոխում են դիտարկվող հատկանիշները (հարձակման ոճը)
    f = fraud
    amount[f] *= rng.uniform(1.0, 1.4, size=f.sum())
    ratio = amount / cust_avg[cid]
    distance[f] += rng.exponential(30, size=f.sum()) * (rng.random(f.sum()) < 0.3)
    tx_1h[f] += rng.poisson(0.3, size=f.sum())

    df = pd.DataFrame({
        "day": day, "hour": hour, "customer_id": cid,
        "amount": amount.round(2), "amount_to_avg_ratio": ratio.round(3),
        "merchant_category": np.array(cats)[cat],
        "tx_count_1h": tx_1h, "distance_from_home_km": distance.round(1),
        "is_foreign": foreign.astype(int), "is_new_device": new_device.astype(int),
        "card_present": card_present.astype(int), "failed_attempts_24h": failed_24h,
        "customer_age": cust_age[cid], "account_age_days": cust_acc_age[cid],
        "is_fraud": fraud.astype(int),
    })
    return df


def _hour_profile():
    p = np.array([1, .6, .4, .3, .3, .5, 1.5, 3, 4.5, 5, 5.5, 6,
                  6.5, 6, 5.5, 5.5, 6, 6.5, 7, 6.5, 5.5, 4, 3, 2], float)
    return p / p.sum()


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/synthetic_transactions.csv", index=False)
    print(d.shape, "fraud rate:", d.is_fraud.mean().round(4))
