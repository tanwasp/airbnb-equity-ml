import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# data was split into 2 csvs bc of github file size limit
df1 = pd.read_csv("data/airbnb_cleaned_part1.csv", index_col=0)
df2 = pd.read_csv("data/airbnb_cleaned_part2.csv", index_col=0)
df = pd.concat([df1, df2], ignore_index=True)

print("shape:", df.shape)

# check missingness for the cols we actually use
key_cols = ["price", "estimated_occupancy_l365d", "estimated_revenue_l365d",
            "review_scores_rating", "median_household_income", "poverty_rate",
            "pct_white", "pct_black", "pct_hispanic", "pct_college_educated"]

miss = df[key_cols].isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
print("\nmissing values in key cols:")
print(pd.DataFrame({"count": miss, "pct": miss_pct}).sort_values("pct", ascending=False))

print("\nsummary stats:")
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

# save combined version so other scripts dont have to re-concat
df.to_csv("progress_report/airbnb_full.csv", index=False)
print("\nsaved combined csv, shape:", df.shape)
