import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, classification_report, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

OUT = "progress_report"
SEED = 395

df = pd.read_csv("progress_report/airbnb_full.csv", low_memory=False)

# negative median age = census missing code
df["median_age"] = df["median_age"].where(df["median_age"] > 0, other=np.nan)

# host_response_rate / host_acceptance_rate are stored as strings like "95%"
for c in ["host_response_rate", "host_acceptance_rate"]:
    df[c] = df[c].astype(str).str.rstrip("%").replace("nan", np.nan)
    df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

# bool cols come in as a mess - sometimes True/False, sometimes "t"/"f", sometimes 1/0
bool_cols = ["host_identity_verified", "host_has_profile_pic",
             "instant_bookable", "host_is_superhost"]
for c in bool_cols:
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0,
                       "t": 1, "f": 0, 1: 1, 0: 0})
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)


# feature lists
listing_features = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "minimum_nights", "maximum_nights",
    "availability_365", "availability_30",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating", "review_scores_cleanliness",
    "review_scores_location", "review_scores_value",
    "host_response_time", "host_response_rate", "host_acceptance_rate",
    "calculated_host_listings_count",
    "has_wifi", "has_parking", "has_pool", "has_gym",
    "instant_bookable", "host_is_superhost",
    "host_identity_verified", "host_has_profile_pic"
]

demo_features = [
    "median_household_income", "median_gross_rent", "median_home_value",
    "total_population", "median_age", "poverty_rate",
    "pct_renter_occupied", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "pct_college_educated",
    "unemployment_rate"
]

# one-hot for room type and property type. property_type has lots of categories
# so we just keep the top 5 and put everything else in "other"
room_dummies = pd.get_dummies(df["room_type"], prefix="room", drop_first=True)
top_props = df["property_type"].value_counts().head(5).index
df["property_type_grouped"] = df["property_type"].where(df["property_type"].isin(top_props), "other")
prop_dummies = pd.get_dummies(df["property_type_grouped"], prefix="prop", drop_first=True)

feat = pd.concat([df[listing_features + demo_features], room_dummies, prop_dummies], axis=1)
# bools -> int for sklearn
bcols = feat.select_dtypes(include=["bool"]).columns
feat[bcols] = feat[bcols].astype(int)

print("total features:", feat.shape[1])


# ---------- Task 1: price regression ----------
print("\n=== price prediction ===")

price_mask = df["price"].notna() & (df["price"] > 0)
X = feat[price_mask].copy()
y = np.log1p(df.loc[price_mask, "price"])  # log helps a lot since prices are right-skewed

# only keep complete cases for now
ok = X.notna().all(axis=1)
X = X[ok]
y = y[ok]
print("samples:", len(X))

# fill any sneaky NaNs with the column median (should be ~0)
X = X.fillna(X.median())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

# scale for the linear model
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

# linear regression baseline
lr = LinearRegression()
lr.fit(X_tr_s, y_tr)
y_pred_lr = lr.predict(X_te_s)

y_te_actual = np.expm1(y_te)
y_pred_lr_actual = np.expm1(y_pred_lr)

rmse_lr = np.sqrt(mean_squared_error(y_te_actual, y_pred_lr_actual))
mae_lr = mean_absolute_error(y_te_actual, y_pred_lr_actual)
r2_lr = r2_score(y_te, y_pred_lr)

print(f"linear: R2={r2_lr:.4f}  RMSE=${rmse_lr:.2f}  MAE=${mae_lr:.2f}")

coef_df = pd.DataFrame({"feature": X_tr.columns, "coefficient": lr.coef_})
coef_df = coef_df.sort_values("coefficient", key=abs, ascending=False)
print("\ntop 10 LR coefficients:")
print(coef_df.head(10).to_string(index=False))

# random forest
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10,
                               random_state=SEED, n_jobs=-1)
rf_reg.fit(X_tr, y_tr)
y_pred_rf = rf_reg.predict(X_te)
y_pred_rf_actual = np.expm1(y_pred_rf)

rmse_rf = np.sqrt(mean_squared_error(y_te_actual, y_pred_rf_actual))
mae_rf = mean_absolute_error(y_te_actual, y_pred_rf_actual)
r2_rf = r2_score(y_te, y_pred_rf)

print(f"\nrandom forest: R2={r2_rf:.4f}  RMSE=${rmse_rf:.2f}  MAE=${mae_rf:.2f}")

importance_df = pd.DataFrame({"feature": X_tr.columns,
                              "importance": rf_reg.feature_importances_})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\ntop 15 RF features:")
print(importance_df.head(15).to_string(index=False))


# ablation: how much do the demographics actually help?
print("\nablation - listing only vs listing+demo")
listing_only = [c for c in X_tr.columns if c not in demo_features]
rf_l = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=10,
                             random_state=SEED, n_jobs=-1)
rf_l.fit(X_tr[listing_only], y_tr)
y_pred_l = rf_l.predict(X_te[listing_only])
r2_listing = r2_score(y_te, y_pred_l)
rmse_listing = np.sqrt(mean_squared_error(y_te_actual, np.expm1(y_pred_l)))
print(f"  listing only: R2={r2_listing:.4f}  RMSE=${rmse_listing:.2f}")
print(f"  listing+demo: R2={r2_rf:.4f}  RMSE=${rmse_rf:.2f}")
print(f"  delta R2: {r2_rf - r2_listing:.4f}")


# ---------- Task 2: high vs low occupancy ----------
print("\n=== occupancy classification ===")

med_occ = df["estimated_occupancy_l365d"].median()
df["high_occupancy"] = (df["estimated_occupancy_l365d"] > med_occ).astype(int)
print(f"median occupancy = {med_occ} nights, "
      f"{df['high_occupancy'].mean()*100:.1f}% above median")

X_o = feat.copy()
y_o = df["high_occupancy"]
ok_o = X_o.notna().all(axis=1) & y_o.notna()
X_o = X_o[ok_o]
y_o = y_o[ok_o]
X_o = X_o.fillna(X_o.median())
print("samples:", len(X_o))

X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
    X_o, y_o, test_size=0.2, random_state=SEED, stratify=y_o)

sc_o = StandardScaler()
X_tr_o_s = sc_o.fit_transform(X_tr_o)
X_te_o_s = sc_o.transform(X_te_o)

# logistic regression baseline
log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
log_reg.fit(X_tr_o_s, y_tr_o)
y_pred_log = log_reg.predict(X_te_o_s)
y_prob_log = log_reg.predict_proba(X_te_o_s)[:, 1]
acc_log = accuracy_score(y_te_o, y_pred_log)
auc_log = roc_auc_score(y_te_o, y_prob_log)
print(f"logistic: acc={acc_log:.4f}  AUC={auc_log:.4f}")
print(classification_report(y_te_o, y_pred_log, target_names=["Low", "High"]))

coef_occ = pd.DataFrame({"feature": X_tr_o.columns, "coefficient": log_reg.coef_[0]})
coef_occ = coef_occ.sort_values("coefficient", key=abs, ascending=False)
print("top 10 LR coefs (occupancy):")
print(coef_occ.head(10).to_string(index=False))

# random forest classifier
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                random_state=SEED, n_jobs=-1)
rf_clf.fit(X_tr_o, y_tr_o)
y_pred_clf = rf_clf.predict(X_te_o)
y_prob_clf = rf_clf.predict_proba(X_te_o)[:, 1]
acc_rf = accuracy_score(y_te_o, y_pred_clf)
auc_rf = roc_auc_score(y_te_o, y_prob_clf)
print(f"\nrandom forest: acc={acc_rf:.4f}  AUC={auc_rf:.4f}")
print(classification_report(y_te_o, y_pred_clf, target_names=["Low", "High"]))

imp_occ_df = pd.DataFrame({"feature": X_tr_o.columns,
                           "importance": rf_clf.feature_importances_})
imp_occ_df = imp_occ_df.sort_values("importance", ascending=False)
print("top 15 RF features (occupancy):")
print(imp_occ_df.head(15).to_string(index=False))


# ablation again
print("\nablation - occupancy")
listing_only_o = [c for c in X_tr_o.columns if c not in demo_features]
rf_clf_l = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                  random_state=SEED, n_jobs=-1)
rf_clf_l.fit(X_tr_o[listing_only_o], y_tr_o)
y_pred_lo = rf_clf_l.predict(X_te_o[listing_only_o])
y_prob_lo = rf_clf_l.predict_proba(X_te_o[listing_only_o])[:, 1]
acc_listing = accuracy_score(y_te_o, y_pred_lo)
auc_listing = roc_auc_score(y_te_o, y_prob_lo)
print(f"  listing only: acc={acc_listing:.4f}  AUC={auc_listing:.4f}")
print(f"  listing+demo: acc={acc_rf:.4f}  AUC={auc_rf:.4f}")
print(f"  delta acc={acc_rf - acc_listing:.4f}  delta AUC={auc_rf - auc_listing:.4f}")


# ---------- plots ----------

# fig 7 - model comparison bars
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(["Linear\nRegression", "Random\nForest"], [r2_lr, r2_rf],
            color=["#4A90D9", "#2ECC71"])
axes[0].set_ylabel("R² Score")
axes[0].set_title("Price Prediction: Model Comparison")
axes[0].set_ylim(0, 1)
for i, v in enumerate([r2_lr, r2_rf]):
    axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

axes[1].bar(["Logistic\nRegression", "Random\nForest"], [auc_log, auc_rf],
            color=["#4A90D9", "#2ECC71"])
axes[1].set_ylabel("AUC Score")
axes[1].set_title("Occupancy Classification: Model Comparison")
axes[1].set_ylim(0, 1)
for i, v in enumerate([auc_log, auc_rf]):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nsaved fig7")


# fig 8 - feature importances side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

t15 = importance_df.head(15)
axes[0].barh(range(len(t15)), t15["importance"].values, color="steelblue")
axes[0].set_yticks(range(len(t15)))
axes[0].set_yticklabels(t15["feature"].values)
axes[0].invert_yaxis()
axes[0].set_xlabel("Feature Importance")
axes[0].set_title("Price Prediction: Top 15 Features (RF)")

t15o = imp_occ_df.head(15)
axes[1].barh(range(len(t15o)), t15o["importance"].values, color="coral")
axes[1].set_yticks(range(len(t15o)))
axes[1].set_yticklabels(t15o["feature"].values)
axes[1].invert_yaxis()
axes[1].set_xlabel("Feature Importance")
axes[1].set_title("Occupancy Classification: Top 15 Features (RF)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig8")


# fig 9 - actual vs predicted scatter
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
idx = np.random.RandomState(SEED).choice(len(y_te_actual), size=min(3000, len(y_te_actual)), replace=False)

axes[0].scatter(y_te_actual.values[idx], y_pred_lr_actual[idx],
                alpha=0.2, s=5, color="steelblue")
axes[0].plot([0, 800], [0, 800], "r--", linewidth=1)
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title(f"Linear Regression (R² = {r2_lr:.3f})")
axes[0].set_xlim(0, 800)
axes[0].set_ylim(0, 800)

axes[1].scatter(y_te_actual.values[idx], y_pred_rf_actual[idx],
                alpha=0.2, s=5, color="#2ECC71")
axes[1].plot([0, 800], [0, 800], "r--", linewidth=1)
axes[1].set_xlabel("Actual Price ($)")
axes[1].set_ylabel("Predicted Price ($)")
axes[1].set_title(f"Random Forest (R² = {r2_rf:.3f})")
axes[1].set_xlim(0, 800)
axes[1].set_ylim(0, 800)

plt.tight_layout()
plt.savefig(f"{OUT}/fig9_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig9")


# fig 10 - ablation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ["Listing Only", "Listing +\nDemographics"]
colors = ["#95a5a6", "#2ECC71"]

vals = [r2_listing, r2_rf]
bars = axes[0].bar(labels, vals, color=colors)
axes[0].set_ylabel("R² Score")
axes[0].set_title("Price Prediction: Effect of Demographics")
axes[0].set_ylim(0, max(vals) * 1.2)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.4f}",
                 ha="center", fontweight="bold")

vals = [auc_listing, auc_rf]
bars = axes[1].bar(labels, vals, color=colors)
axes[1].set_ylabel("AUC Score")
axes[1].set_title("Occupancy Classification: Effect of Demographics")
axes[1].set_ylim(0, max(vals) * 1.2)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.4f}",
                 ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/fig10_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig10")


# write a summary so we can copy into the report
with open(f"{OUT}/results_summary.txt", "w") as f:
    f.write("PRICE PREDICTION\n")
    f.write(f"Linear Regression: R2 = {r2_lr:.4f}, RMSE = ${rmse_lr:.2f}, MAE = ${mae_lr:.2f}\n")
    f.write(f"Random Forest:     R2 = {r2_rf:.4f}, RMSE = ${rmse_rf:.2f}, MAE = ${mae_rf:.2f}\n")
    f.write(f"Listing only RF:   R2 = {r2_listing:.4f}, RMSE = ${rmse_listing:.2f}\n")
    f.write(f"Demo improvement:  delta R2 = {r2_rf - r2_listing:.4f}\n\n")
    f.write("OCCUPANCY CLASSIFICATION\n")
    f.write(f"Logistic Regression: acc = {acc_log:.4f}, AUC = {auc_log:.4f}\n")
    f.write(f"Random Forest:       acc = {acc_rf:.4f}, AUC = {auc_rf:.4f}\n")
    f.write(f"Listing only RF:     acc = {acc_listing:.4f}, AUC = {auc_listing:.4f}\n")
    f.write(f"Demo improvement:    delta acc = {acc_rf - acc_listing:.4f}, delta AUC = {auc_rf - auc_listing:.4f}\n\n")
    f.write("TOP FEATURES (PRICE - RF):\n")
    f.write(importance_df.head(15).to_string(index=False) + "\n\n")
    f.write("TOP FEATURES (OCCUPANCY - RF):\n")
    f.write(imp_occ_df.head(15).to_string(index=False) + "\n")

print("\ndone, results_summary.txt written")
