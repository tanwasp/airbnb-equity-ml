# per-city analysis - do demographics matter equally everywhere?
# fit RF price models within each of the 8 biggest cities and compare.
# also pull a standardized linear coefficient on pct_black per city.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

SEED = 395

print("loading...")
df = pd.read_csv("progress_report/airbnb_full.csv", low_memory=False)

# same cleaning as everywhere else
df["median_age"] = df["median_age"].where(df["median_age"] > 0, other=np.nan)
for c in ["host_response_rate", "host_acceptance_rate"]:
    df[c] = df[c].astype(str).str.rstrip("%").replace("nan", np.nan)
    df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
for c in ["host_identity_verified", "host_has_profile_pic",
          "instant_bookable", "host_is_superhost"]:
    df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0,
                       "t": 1, "f": 0, 1: 1, 0: 0})
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

# slimmer feature list bc per-city sample sizes are smaller
listing_features = [
    "accommodates", "bedrooms", "beds", "bathrooms",
    "minimum_nights", "availability_365", "availability_30",
    "number_of_reviews", "reviews_per_month",
    "review_scores_rating",
    "host_response_rate", "host_acceptance_rate",
    "has_wifi", "has_parking", "has_pool", "has_gym",
    "instant_bookable", "host_is_superhost"
]
demo_features = [
    "median_household_income", "median_gross_rent", "median_home_value",
    "poverty_rate", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "pct_college_educated",
    "unemployment_rate"
]

room_dummies = pd.get_dummies(df["room_type"], prefix="room", drop_first=True)

top_cities = df["city"].value_counts().head(8).index.tolist()
print("cities:", top_cities)

results = []
for city in top_cities:
    sub = df[df["city"] == city].copy()
    sub = sub[sub["price"].notna() & (sub["price"] > 0)]
    if len(sub) < 1000:
        continue

    feats = pd.concat([sub[listing_features + demo_features],
                       room_dummies.loc[sub.index]], axis=1)
    bcols = feats.select_dtypes(include=["bool"]).columns
    feats[bcols] = feats[bcols].astype(int)

    imp = SimpleImputer(strategy="median")
    X = pd.DataFrame(imp.fit_transform(feats), columns=feats.columns, index=feats.index)
    y = np.log1p(sub["price"])

    if len(X) < 500:
        continue

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # full feature set
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=10,
                               random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    r2_full = r2_score(y_te, rf.predict(X_te))

    # listing-only
    listing_cols = [c for c in X_tr.columns if c not in demo_features]
    rf_l = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=10,
                                 random_state=SEED, n_jobs=-1)
    rf_l.fit(X_tr[listing_cols], y_tr)
    r2_listing = r2_score(y_te, rf_l.predict(X_te[listing_cols]))

    # standardized LR coefficient on pct_black (a rough effect size estimate)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_tr)
    lr = LinearRegression()
    lr.fit(Xs, y_tr)
    pb_idx = list(X_tr.columns).index("pct_black")
    pct_black_coef = lr.coef_[pb_idx]

    results.append({
        "city": city,
        "n": len(X),
        "median_price": np.expm1(y).median(),
        "r2_listing_only": r2_listing,
        "r2_with_demo": r2_full,
        "r2_gain_from_demo": r2_full - r2_listing,
        "pct_black_coef_lr": pct_black_coef,
    })
    print(f"  {city}: n={len(X)}, R2={r2_full:.3f}, gain={r2_full-r2_listing:+.4f}, "
          f"pct_black coef={pct_black_coef:+.4f}")

city_df = pd.DataFrame(results).sort_values("n", ascending=False)
print("\nper-city results:")
print(city_df.to_string(index=False))


# plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
y_pos = np.arange(len(city_df))
ax.barh(y_pos, city_df["r2_gain_from_demo"], color="#20808D")
ax.set_yticks(y_pos)
ax.set_yticklabels(city_df["city"])
ax.invert_yaxis()
ax.set_xlabel("Gain in R^2 from adding demographics")
ax.set_title("How much demographics help price prediction (per city)")
ax.axvline(0, color="black", linewidth=0.5)

ax = axes[1]
cols = ["#A84B2F" if c < 0 else "#20808D" for c in city_df["pct_black_coef_lr"]]
ax.barh(y_pos, city_df["pct_black_coef_lr"], color=cols)
ax.set_yticks(y_pos)
ax.set_yticklabels(city_df["city"])
ax.invert_yaxis()
ax.set_xlabel("Standardized LR coefficient on pct_black\n(controlling for listing features)")
ax.set_title("Effect of neighborhood %Black on log price (per city)")
ax.axvline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("progress_report/fig14_city_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig14")

city_df.to_csv("progress_report/city_results.csv", index=False)
print("done")
