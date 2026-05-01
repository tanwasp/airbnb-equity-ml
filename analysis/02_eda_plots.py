import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

OUT = "progress_report"

df = pd.read_csv("progress_report/airbnb_full.csv")

# negative median_age means missing in the census data
df["median_age"] = df["median_age"].where(df["median_age"] > 0, other=np.nan)

# for price plots we need a non-null positive price
df_price = df.dropna(subset=["price"])
df_price = df_price[df_price["price"] > 0]

print("full:", len(df), "with valid price:", len(df_price))


# fig 1 - target distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(df_price["price"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Nightly Price ($)")
axes[0].set_xlabel("Price")
axes[0].set_ylabel("Count")

axes[1].hist(df["estimated_occupancy_l365d"], bins=50, color="steelblue", edgecolor="white")
axes[1].set_title("Estimated Occupancy (last 365 days)")
axes[1].set_xlabel("Nights booked")

df_rev = df.dropna(subset=["estimated_revenue_l365d"])
axes[2].hist(df_rev["estimated_revenue_l365d"], bins=50, color="steelblue", edgecolor="white")
axes[2].set_title("Estimated Revenue (last 365 days)")
axes[2].set_xlabel("Revenue ($)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_target_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig1")


# fig 2 - correlation heatmap
corr_cols = ["price", "estimated_occupancy_l365d", "estimated_revenue_l365d",
             "review_scores_rating", "accommodates", "bedrooms", "beds",
             "number_of_reviews", "reviews_per_month",
             "median_household_income", "poverty_rate", "pct_renter_occupied",
             "pct_white", "pct_black", "pct_hispanic", "pct_college_educated",
             "unemployment_rate"]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))  # only show lower triangle
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            annot_kws={"size": 7})
ax.set_title("Correlation Matrix: Listing and Demographic Features")
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig2")


# fig 3 - price + occupancy by room type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
room_order = df_price["room_type"].value_counts().index.tolist()
sns.boxplot(data=df_price, x="room_type", y="price", order=room_order, ax=axes[0])
axes[0].set_title("Price by Room Type")
axes[0].set_xlabel("")
axes[0].set_ylabel("Price ($)")

sns.boxplot(data=df, x="room_type", y="estimated_occupancy_l365d", order=room_order, ax=axes[1])
axes[1].set_title("Occupancy by Room Type")
axes[1].set_xlabel("")
axes[1].set_ylabel("Estimated Occupancy (nights)")

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_room_type.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig3")


# fig 4 - demographic features vs price
demo_cols = ["median_household_income", "poverty_rate", "pct_white",
             "pct_black", "pct_college_educated", "unemployment_rate"]
demo_labels = ["Median Household Income", "Poverty Rate", "% White",
               "% Black", "% College Educated", "Unemployment Rate"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, col, label in zip(axes.flatten(), demo_cols, demo_labels):
    # subsample so the scatter isnt a solid blob
    samp = df_price.dropna(subset=[col]).sample(min(5000, len(df_price)), random_state=395)
    ax.scatter(samp[col], samp["price"], alpha=0.15, s=5, color="steelblue")
    ax.set_xlabel(label)
    ax.set_ylabel("Price ($)")
    r = df_price[[col, "price"]].dropna().corr().iloc[0, 1]
    ax.set_title(f"{label}\nr = {r:.3f}")

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_demographics_vs_price.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig4")


# fig 5 - top cities
fig, ax = plt.subplots(figsize=(10, 5))
df["city"].value_counts().head(15).plot(kind="barh", color="steelblue", ax=ax)
ax.set_xlabel("Number of Listings")
ax.set_title("Top 15 Cities by Number of Listings")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_city_counts.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig5")


# fig 6 - demographics vs occupancy (same idea as fig4 but for occupancy)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, col, label in zip(axes.flatten(), demo_cols, demo_labels):
    samp = df.dropna(subset=[col]).sample(min(5000, len(df)), random_state=395)
    ax.scatter(samp[col], samp["estimated_occupancy_l365d"], alpha=0.15, s=5, color="coral")
    ax.set_xlabel(label)
    ax.set_ylabel("Occupancy (nights)")
    r = df[[col, "estimated_occupancy_l365d"]].dropna().corr().iloc[0, 1]
    ax.set_title(f"{label}\nr = {r:.3f}")

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_demographics_vs_occupancy.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved fig6")

print("done")
