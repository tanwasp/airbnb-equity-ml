# gradient boosting (xgb + lgbm) for both tasks + a small grid search.
# from script 04 we picked median imputation since it kept all rows
# without much loss in R2.

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

SEED = 395
OUT = "progress_report"

print("loading...")
df = pd.read_csv("progress_report/airbnb_full.csv", low_memory=False)

# same cleaning
df["median_age"] = df["median_age"].where(df["median_age"] > 0, other=np.nan)
for c in ["host_response_rate", "host_acceptance_rate"]:
    df[c] = df[c].astype(str).str.rstrip("%").replace("nan", np.nan)
    df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
for c in ["host_identity_verified", "host_has_profile_pic",
          "instant_bookable", "host_is_superhost"]:
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0,
                       "t": 1, "f": 0, 1: 1, 0: 0})
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

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

room_dummies = pd.get_dummies(df["room_type"], prefix="room", drop_first=True)
top_props = df["property_type"].value_counts().head(5).index
df["property_type_grouped"] = df["property_type"].where(df["property_type"].isin(top_props), "other")
prop_dummies = pd.get_dummies(df["property_type_grouped"], prefix="prop", drop_first=True)

feat = pd.concat([df[listing_features + demo_features], room_dummies, prop_dummies], axis=1)
bcols = feat.select_dtypes(include=["bool"]).columns
feat[bcols] = feat[bcols].astype(int)


# ---- price ----
print("\n--- price ---")

price_mask = df["price"].notna() & (df["price"] > 0)
X_price = feat[price_mask].copy()
y_price = np.log1p(df.loc[price_mask, "price"])

imp = SimpleImputer(strategy="median")
X_price_imp = pd.DataFrame(imp.fit_transform(X_price), columns=X_price.columns,
                           index=X_price.index)
print("samples:", len(X_price_imp))

X_tr, X_te, y_tr, y_te = train_test_split(X_price_imp, y_price, test_size=0.2, random_state=SEED)
y_te_actual = np.expm1(y_te)

# xgboost (default-ish)
print("xgboost...")
xgb_reg = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           random_state=SEED, n_jobs=-1)
xgb_reg.fit(X_tr, y_tr)
yp = xgb_reg.predict(X_te)
r2_xgb_p = r2_score(y_te, yp)
rmse_xgb_p = np.sqrt(mean_squared_error(y_te_actual, np.expm1(yp)))
mae_xgb_p = mean_absolute_error(y_te_actual, np.expm1(yp))
print(f"  R2={r2_xgb_p:.4f}  RMSE=${rmse_xgb_p:.2f}  MAE=${mae_xgb_p:.2f}")

# lightgbm
print("lightgbm...")
lgb_reg = lgb.LGBMRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, verbose=-1)
lgb_reg.fit(X_tr, y_tr)
yp = lgb_reg.predict(X_te)
r2_lgb_p = r2_score(y_te, yp)
rmse_lgb_p = np.sqrt(mean_squared_error(y_te_actual, np.expm1(yp)))
mae_lgb_p = mean_absolute_error(y_te_actual, np.expm1(yp))
print(f"  R2={r2_lgb_p:.4f}  RMSE=${rmse_lgb_p:.2f}  MAE=${mae_lgb_p:.2f}")

# small grid search on a 40k subsample (full grid would take forever)
print("xgb grid search (subsample for speed)...")
n_tune = min(40000, len(X_tr))
X_tune = X_tr.sample(n=n_tune, random_state=SEED)
y_tune = y_tr.loc[X_tune.index]

param_grid = {
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 400],
}
xgb_base = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1)
grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring="r2", n_jobs=2)
grid.fit(X_tune, y_tune)
print("  best:", grid.best_params_, "CV R2:", round(grid.best_score_, 4))

# refit best model on the full training data
best_xgb = grid.best_estimator_
best_xgb.fit(X_tr, y_tr)
yp = best_xgb.predict(X_te)
r2_best_p = r2_score(y_te, yp)
rmse_best_p = np.sqrt(mean_squared_error(y_te_actual, np.expm1(yp)))
print(f"  tuned full-train: R2={r2_best_p:.4f}  RMSE=${rmse_best_p:.2f}")

# 5-fold cv on a 50k subsample
print("5-fold cv (50k subsample)...")
idx = np.random.RandomState(SEED).choice(len(X_price_imp), size=min(50000, len(X_price_imp)), replace=False)
X_cv = X_price_imp.iloc[idx]
y_cv = y_price.iloc[idx]
cv_scores = cross_val_score(
    xgb.XGBRegressor(**grid.best_params_, subsample=0.8, colsample_bytree=0.8,
                     random_state=SEED, n_jobs=-1),
    X_cv, y_cv, cv=5, scoring="r2", n_jobs=2)
print("  fold R2s:", [round(s, 4) for s in cv_scores])
print(f"  mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")


# ---- occupancy ----
print("\n--- occupancy ---")
med_occ = df["estimated_occupancy_l365d"].median()
df["high_occupancy"] = (df["estimated_occupancy_l365d"] > med_occ).astype(int)

X_o = feat.copy()
y_o = df["high_occupancy"]
ok = y_o.notna()
X_o = X_o[ok]; y_o = y_o[ok]

imp_o = SimpleImputer(strategy="median")
X_o_imp = pd.DataFrame(imp_o.fit_transform(X_o), columns=X_o.columns, index=X_o.index)
print("samples:", len(X_o_imp))

X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
    X_o_imp, y_o, test_size=0.2, random_state=SEED, stratify=y_o)

print("xgboost...")
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=-1, eval_metric="logloss")
xgb_clf.fit(X_tr_o, y_tr_o)
yp_o = xgb_clf.predict(X_te_o)
yprob_o = xgb_clf.predict_proba(X_te_o)[:, 1]
acc_xgb_o = accuracy_score(y_te_o, yp_o)
auc_xgb_o = roc_auc_score(y_te_o, yprob_o)
print(f"  acc={acc_xgb_o:.4f}  AUC={auc_xgb_o:.4f}")

print("lightgbm...")
lgb_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                             num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=-1, verbose=-1)
lgb_clf.fit(X_tr_o, y_tr_o)
yp_o = lgb_clf.predict(X_te_o)
yprob_o = lgb_clf.predict_proba(X_te_o)[:, 1]
acc_lgb_o = accuracy_score(y_te_o, yp_o)
auc_lgb_o = roc_auc_score(y_te_o, yprob_o)
print(f"  acc={acc_lgb_o:.4f}  AUC={auc_lgb_o:.4f}")

print("5-fold cv (50k subsample)...")
idx_o = np.random.RandomState(SEED).choice(len(X_o_imp), size=min(50000, len(X_o_imp)), replace=False)
X_cv_o = X_o_imp.iloc[idx_o]
y_cv_o = y_o.iloc[idx_o]
cv_scores_o = cross_val_score(
    xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      random_state=SEED, n_jobs=-1, eval_metric="logloss"),
    X_cv_o, y_cv_o, cv=5, scoring="roc_auc", n_jobs=2)
print("  fold AUCs:", [round(s, 4) for s in cv_scores_o])
print(f"  mean={cv_scores_o.mean():.4f}  std={cv_scores_o.std():.4f}")


# pickle the models so the SHAP / fairness script can reuse them
with open(f"{OUT}/models.pkl", "wb") as f:
    pickle.dump({
        "best_xgb_price": best_xgb,
        "xgb_clf_occ": xgb_clf,
        "X_te_price": X_te,
        "y_te_price": y_te,
        "X_te_occ": X_te_o,
        "y_te_occ": y_te_o,
        "feature_cols": list(X_price_imp.columns),
    }, f)

with open(f"{OUT}/advanced_results.txt", "w") as f:
    f.write("PRICE (median-imputed)\n")
    f.write(f"XGBoost:        R2={r2_xgb_p:.4f}  RMSE=${rmse_xgb_p:.2f}  MAE=${mae_xgb_p:.2f}\n")
    f.write(f"LightGBM:       R2={r2_lgb_p:.4f}  RMSE=${rmse_lgb_p:.2f}  MAE=${mae_lgb_p:.2f}\n")
    f.write(f"XGBoost tuned:  R2={r2_best_p:.4f}  RMSE=${rmse_best_p:.2f}\n")
    f.write(f"best params: {grid.best_params_}\n")
    f.write(f"5-fold CV R2: mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}\n")
    f.write(f"  folds: {list(cv_scores)}\n\n")
    f.write("OCCUPANCY\n")
    f.write(f"XGBoost:  acc={acc_xgb_o:.4f}  AUC={auc_xgb_o:.4f}\n")
    f.write(f"LightGBM: acc={acc_lgb_o:.4f}  AUC={auc_lgb_o:.4f}\n")
    f.write(f"5-fold CV AUC: mean={cv_scores_o.mean():.4f}  std={cv_scores_o.std():.4f}\n")
    f.write(f"  folds: {list(cv_scores_o)}\n")

print("\ndone")
