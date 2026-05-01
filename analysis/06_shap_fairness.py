# SHAP for the price model + fairness metrics for the occupancy classifier
# (demographic parity / equalized odds across racial-composition and income groups).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import warnings
warnings.filterwarnings("ignore")

OUT = "progress_report"
SEED = 395

print("loading models + data...")
with open(f"{OUT}/models.pkl", "rb") as f:
    saved = pickle.load(f)

best_xgb_price = saved["best_xgb_price"]
xgb_clf_occ = saved["xgb_clf_occ"]
X_te_price = saved["X_te_price"]
X_te_occ = saved["X_te_occ"]
y_te_occ = saved["y_te_occ"]

df = pd.read_csv(f"{OUT}/airbnb_full.csv", low_memory=False)


# ---- SHAP on price ----
print("\n--- SHAP (price) ---")

# 2000 row sample - full test set is way too slow with shap.TreeExplainer
n_shap = 2000
idx = np.random.RandomState(SEED).choice(len(X_te_price), size=n_shap, replace=False)
X_shap = X_te_price.iloc[idx]

expl = shap.TreeExplainer(best_xgb_price)
shap_vals = expl.shap_values(X_shap)

shap_imp = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("top 15 by mean |SHAP|:")
print(shap_imp.head(15).to_string(index=False))

shap_imp.to_csv(f"{OUT}/shap_importance_price.csv", index=False)

# bar chart
fig, ax = plt.subplots(figsize=(8, 7))
top15 = shap_imp.head(15).iloc[::-1]
ax.barh(top15["feature"], top15["mean_abs_shap"], color="#20808D")
ax.set_xlabel("Mean |SHAP value| (impact on log price)")
ax.set_title("Feature importance for price (XGBoost, SHAP)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_shap_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig11")

# beeswarm - useful but kind of busy
plt.figure()
shap.summary_plot(shap_vals, X_shap, max_display=15, show=False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig12_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig12")

# how much of the model's |shap| is demographics?
demo_features = [
    "median_household_income", "median_gross_rent", "median_home_value",
    "total_population", "median_age", "poverty_rate",
    "pct_renter_occupied", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "pct_college_educated",
    "unemployment_rate"
]
demo_in_features = [f for f in demo_features if f in X_shap.columns]
total_abs = np.abs(shap_vals).sum()
demo_idx_list = [list(X_shap.columns).index(f) for f in demo_in_features]
demo_abs = np.abs(shap_vals[:, demo_idx_list]).sum()
demo_share = demo_abs / total_abs * 100
print(f"\ndemographics = {demo_share:.1f}% of total |SHAP| in price model")


# ---- fairness on occupancy ----
print("\n--- fairness (occupancy) ---")

y_pred_o = xgb_clf_occ.predict(X_te_occ)
y_prob_o = xgb_clf_occ.predict_proba(X_te_occ)[:, 1]

# pull the demographic columns from the original df, lined up with the test rows
test_demos = df.loc[X_te_occ.index, ["pct_black", "pct_white", "pct_hispanic",
                                      "median_household_income", "city"]].copy()
test_demos["y_true"] = y_te_occ.values
test_demos["y_pred"] = y_pred_o
test_demos["y_prob"] = y_prob_o

# group A: top quartile of pct_black ("high-Black tracts") vs the rest
thr_b = test_demos["pct_black"].quantile(0.75)
test_demos["high_black_nbhd"] = (test_demos["pct_black"] > thr_b).astype(int)

# group B: bottom quartile of median income ("low-income tracts") vs the rest
thr_i = test_demos["median_household_income"].quantile(0.25)
test_demos["low_income_nbhd"] = (test_demos["median_household_income"] < thr_i).astype(int)


def fairness_summary(grp_col):
    # selection rate, TPR, FPR per group
    rows = []
    for grp_val, grp in test_demos.groupby(grp_col):
        sel = grp["y_pred"].mean()
        tpr = grp.loc[grp["y_true"] == 1, "y_pred"].mean() if (grp["y_true"] == 1).any() else np.nan
        fpr = grp.loc[grp["y_true"] == 0, "y_pred"].mean() if (grp["y_true"] == 0).any() else np.nan
        rows.append({
            "group": f"{grp_col}={grp_val}",
            "n": len(grp),
            "selection_rate": sel,
            "TPR": tpr,
            "FPR": fpr,
            "true_pos_rate": grp["y_true"].mean(),
        })
    return pd.DataFrame(rows)


def gap(df_, grp_col, metric):
    a = df_.loc[df_["group"] == f"{grp_col}=1", metric].iloc[0]
    b = df_.loc[df_["group"] == f"{grp_col}=0", metric].iloc[0]
    return abs(a - b)


print("\nby neighborhood race composition (high_black = top-quartile pct_black):")
fb = fairness_summary("high_black_nbhd")
print(fb.to_string(index=False))
dp_b = gap(fb, "high_black_nbhd", "selection_rate")
tpr_b = gap(fb, "high_black_nbhd", "TPR")
fpr_b = gap(fb, "high_black_nbhd", "FPR")
print(f"DP gap = {dp_b:.4f}, TPR gap = {tpr_b:.4f}, FPR gap = {fpr_b:.4f}")

print("\nby neighborhood income (low_income = bottom-quartile median income):")
fi = fairness_summary("low_income_nbhd")
print(fi.to_string(index=False))
dp_i = gap(fi, "low_income_nbhd", "selection_rate")
tpr_i = gap(fi, "low_income_nbhd", "TPR")
fpr_i = gap(fi, "low_income_nbhd", "FPR")
print(f"DP gap = {dp_i:.4f}, TPR gap = {tpr_i:.4f}, FPR gap = {fpr_i:.4f}")


# fairness bar plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# race
ax = axes[0]
metrics = {
    "Selection rate": [fb.loc[fb["group"] == "high_black_nbhd=0", "selection_rate"].iloc[0],
                       fb.loc[fb["group"] == "high_black_nbhd=1", "selection_rate"].iloc[0]],
    "TPR": [fb.loc[fb["group"] == "high_black_nbhd=0", "TPR"].iloc[0],
            fb.loc[fb["group"] == "high_black_nbhd=1", "TPR"].iloc[0]],
    "FPR": [fb.loc[fb["group"] == "high_black_nbhd=0", "FPR"].iloc[0],
            fb.loc[fb["group"] == "high_black_nbhd=1", "FPR"].iloc[0]],
}
x = np.arange(len(metrics))
w = 0.35
ax.bar(x - w/2, [v[0] for v in metrics.values()], w, label="Other tracts", color="#95a5a6")
ax.bar(x + w/2, [v[1] for v in metrics.values()], w, label="High-Black tracts", color="#A84B2F")
ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()))
ax.set_ylabel("Rate"); ax.set_title("Fairness by neighborhood race composition")
ax.legend(); ax.set_ylim(0, 1)

# income
ax = axes[1]
metrics = {
    "Selection rate": [fi.loc[fi["group"] == "low_income_nbhd=0", "selection_rate"].iloc[0],
                       fi.loc[fi["group"] == "low_income_nbhd=1", "selection_rate"].iloc[0]],
    "TPR": [fi.loc[fi["group"] == "low_income_nbhd=0", "TPR"].iloc[0],
            fi.loc[fi["group"] == "low_income_nbhd=1", "TPR"].iloc[0]],
    "FPR": [fi.loc[fi["group"] == "low_income_nbhd=0", "FPR"].iloc[0],
            fi.loc[fi["group"] == "low_income_nbhd=1", "FPR"].iloc[0]],
}
ax.bar(x - w/2, [v[0] for v in metrics.values()], w, label="Higher-income", color="#95a5a6")
ax.bar(x + w/2, [v[1] for v in metrics.values()], w, label="Low-income tracts", color="#1B474D")
ax.set_xticks(x); ax.set_xticklabels(list(metrics.keys()))
ax.set_ylabel("Rate"); ax.set_title("Fairness by neighborhood income")
ax.legend(); ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUT}/fig13_fairness.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig13")

with open(f"{OUT}/fairness_results.txt", "w") as f:
    f.write("FAIRNESS METRICS (occupancy classifier)\n\n")
    f.write("by neighborhood race composition (high_black = top quartile):\n")
    f.write(fb.to_string(index=False) + "\n")
    f.write(f"DP gap: {dp_b:.4f}  TPR gap: {tpr_b:.4f}  FPR gap: {fpr_b:.4f}\n\n")
    f.write("by neighborhood income (low_income = bottom quartile):\n")
    f.write(fi.to_string(index=False) + "\n")
    f.write(f"DP gap: {dp_i:.4f}  TPR gap: {tpr_i:.4f}  FPR gap: {fpr_i:.4f}\n\n")
    f.write(f"demographics share of total |SHAP| in price model: {demo_share:.1f}%\n")

print("\ndone")
