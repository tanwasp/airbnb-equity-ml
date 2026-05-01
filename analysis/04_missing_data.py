# trying out a few ways to deal with the missing values, since a lot
# of price/review_score rows are NaN. compare RF performance after each.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

SEED = 395

print("loading...")
df = pd.read_csv("progress_report/airbnb_full.csv", low_memory=False)
print("rows:", len(df))

# same cleaning as before
df["median_age"] = df["median_age"].where(df["median_age"] > 0, other=np.nan)
for c in ["host_response_rate", "host_acceptance_rate"]:
    df[c] = df[c].astype(str).str.rstrip("%").replace("nan", np.nan)
    df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
for c in ["host_identity_verified", "host_has_profile_pic",
          "instant_bookable", "host_is_superhost"]:
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0,
                       "t": 1, "f": 0, 1: 1, 0: 0})
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

print("\nmissingness:")
for col in ["price", "review_scores_rating", "reviews_per_month"]:
    print(f"  {col}: {df[col].isna().sum()} ({df[col].isna().mean()*100:.1f}%)")

# same feature setup as before
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

# need a non-missing y
price_mask = df["price"].notna() & (df["price"] > 0)
X_all = feat[price_mask].copy()
y_all = np.log1p(df.loc[price_mask, "price"])
print("rows w/ valid price:", len(X_all))


def fit_rf(X, y, label):
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=10,
                               random_state=SEED, n_jobs=-1)
    rf.fit(Xt, yt)
    yp = rf.predict(Xe)
    r2 = r2_score(ye, yp)
    rmse = np.sqrt(mean_squared_error(np.expm1(ye), np.expm1(yp)))
    print(f"  {label}: R2={r2:.4f}  RMSE=${rmse:.2f}")
    return r2, rmse


# strategy 1: drop rows with any missing feature
print("\n[1] drop rows with any missing")
ok = X_all.notna().all(axis=1)
X1 = X_all[ok]; y1 = y_all[ok]
print(f"  rows kept: {len(X1)} ({len(X1)/len(X_all)*100:.1f}%)")
r2_drop, rmse_drop = fit_rf(X1, y1, "drop")

# strategy 2: mean impute
print("\n[2] mean imputation")
imp = SimpleImputer(strategy="mean")
X2 = pd.DataFrame(imp.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
print(f"  rows: {len(X2)}")
r2_mean, rmse_mean = fit_rf(X2, y_all, "mean")

# strategy 3: median impute
# we use median instead of KNN on the full data because KNN takes forever
# on ~280k rows. median is a fine simple baseline.
print("\n[3] median imputation")
imp = SimpleImputer(strategy="median")
X3 = pd.DataFrame(imp.fit_transform(X_all), columns=X_all.columns, index=X_all.index)
print(f"  rows: {len(X3)}")
r2_med, rmse_med = fit_rf(X3, y_all, "median")

# strategy 4: KNN on a 30k subsample (full data is way too slow)
print("\n[4] KNN imputation (k=5, on 30k subsample)")
samp = X_all.sample(n=min(30000, len(X_all)), random_state=SEED)
samp_y = y_all.loc[samp.index]
knn = KNNImputer(n_neighbors=5)
X4 = pd.DataFrame(knn.fit_transform(samp), columns=samp.columns, index=samp.index)
print(f"  rows: {len(X4)}")
r2_knn, rmse_knn = fit_rf(X4, samp_y, "knn")
print("  (different sample size, so direct comparison is rough)")

with open("progress_report/missing_data_results.txt", "w") as f:
    f.write("MISSING DATA STRATEGY COMPARISON\n")
    f.write(f"drop:    R2={r2_drop:.4f}  RMSE=${rmse_drop:.2f}  n={len(X1)}\n")
    f.write(f"mean:    R2={r2_mean:.4f}  RMSE=${rmse_mean:.2f}  n={len(X2)}\n")
    f.write(f"median:  R2={r2_med:.4f}  RMSE=${rmse_med:.2f}  n={len(X3)}\n")
    f.write(f"knn:     R2={r2_knn:.4f}  RMSE=${rmse_knn:.2f}  n={len(X4)}\n")

print("\ndone")
