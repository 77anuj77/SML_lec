import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

df=pd.read_csv("wine_quality.csv")
print(df.head(3))

#bins alpla --> histogram
fig, axes= plt.subplots(1,3, figsize= (14,4))
fig.suptitle("Distribution of key wine Features", fontsize=13, fontweight="bold")

for ax, col in zip(axes, ["alcohol", "color_intensity", "proline"]):
    ax.hist(df[col], bins=20, color= "#028098", edgecolor="white", alpha=0.85)
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

#when to use bar chart and hitogram , multivarate and unovarate 
'''Use a bar chart for comparing categories. Use a histogram for understanding the distribution of continuous numerical data.'''

import matplotlib.pyplot as plt

avg_alcohol = (
    df.groupby("quality_label")["alcohol"].mean().sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    avg_alcohol.index,
    avg_alcohol.values,
    color=["#02C39A", "#028090", "#082545"],
    edgecolor="white",
    width=0.5
)

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{bar.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_title("Average Alcohol Content by Quality Label", fontsize=12)
ax.set_xlabel("Quality Label")
ax.set_ylabel("Mean Alcohol (%)")
ax.set_ylim(0, 15)

plt.tight_layout()
plt.show()

colour_map = {"high": 0, "medium": 1, "low": 2}
colours = df["quality_label"].map(colour_map)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["alcohol"], df["color_intensity"], c=colours, cmap="viridis", alpha=0.7, edgecolors="white", s=60)

ax.set_xlabel("Alcohol (%)", fontsize=11)
ax.set_ylabel("Colour Intensity", fontsize=11)
ax.set_title("Alcohol vs Colour Intensity (coloured by quality)", fontsize=12)

plt.tight_layout()

plt.show()


fig, axes= plt.subplots(1,2,figsize=(13,5))
fig.suptitle("Feature distribution by Quality Label", fontsize=13, fontweight="bold")
order=["high", "medium", "low"]
sns.boxplot(data=df, x="quality_label", y="alcohol", order=order, palette="Set2", ax=axes[0])
axes[0].set_title("Alcohol by Quality")
axes[0].set_xlabel("Quality Label")
axes[0].set_ylabel("Alcohol (%)")

sns.boxplot(data=df, x="quality_label", y="proline", order=order, palette="Set2", ax=axes[1])
axes[1].set_title("Peoline by Quality")
axes[1].set_xlabel("Quality Label")
axes[1].set_ylabel("Proline")

plt.tight_layout()
plt.show()


import numpy as np
#corr is a function uses pearson correlation for the numerical data
numeric_df=df.drop(columns=["quality_label"])
corr=numeric_df.corr()
mask=np.triu(np.ones_like(corr, dtype=bool))
fig, ax=plt.subplots(figsize=(11,8))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn", center=0, linewidths=0.4, annot_kws={"size":8}, ax=ax)
ax.set_title("Feature COrrelation Heatmap - Wine dataset", fontsize=13, pad=12)
plt.tight_layout()
plt.show
df.shape

df.columns

subset_df=df[["alcohol", "flavanoids", "color_intensity", "proline", "quality_label"]]
#quality_label is not include in the pairplot because it is target variable
pair_plot=sns.pairplot(subset_df, hue="quality_label", palette={"high": "green", "medium": "orange", "low": "red"}, diag_kind="hist", plot_kws={"alpha":0.6, "s": 30})
pair_plot.figure.suptitle("Pair Plot -Selected Wine Feature", y=1.01, fontsize=13)
plt.tight_layout()
plt.show()

le=LabelEncoder()
df["quality_le"]=le.fit_transform(df["quality_label"])
print("label Encoder mapping", dict(zip(le.classes_, le.transform(le.classes_))))

order_map={"low": 0, "medium":1, "high":2}
df["quality_label"]=df["quality_label"].map(order_map)
print("Manual ordinal mapping:" , order_map)

df_ohe = pd.get_dummies(df, columns=['quality_label'], drop_first=True)
new_cols = [c for c in df_ohe.columns if "quality_label" in c]

print("OHE new cols:\n", *new_cols)
print(f"Shape before OHE {df.shape} | after: {df_ohe.shape}")

df[["quality_label", "quality_le", "quality_ordinal"]].drop_duplicates().sort_values("quality_ordinal")

df_eng=df.copy()

df_eng["alcohol_acidity_ratio"]= df_eng["alcohol"]/ df_eng["malic_acid"]+1e-6
df_eng["flavonoids_phenols_ratio"]= df_eng["flavanoids"]/ df_eng["total_phenols"]+1e-6
df_eng["color_per_alcohol"]= df_eng["color_intensity"]/ df_eng["alcohol"]+1e-6

# Bin alcohol into categories
df_eng["alcohol_bins"] = pd.cut(df_eng["alcohol"], 3, labels = ["low", "medium", "high"])

# Showing distribution
print(df_eng["alcohol_bins"].value_counts())

# Compare alcohol vs quality
pd.crosstab(df_eng["alcohol_bins"], df_eng["quality_label"])


corr= df_eng.corr(numeric_only=True).abs()
high_corr=corr.index[(corr>0.85).sum()>1].tolist()
print("Highly correlated features to check: ")
print(high_corr)

#PCA principal componenet analysis
#pc1 captures the most variance

#pc3 aptures the second most and so on 

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X= df_eng.select_dtypes(include=[np.number]).dropna()

scaler=StandardScaler()
X_scaled= scaler.fit_transform(X)

pca=PCA(n_components=4)
X_pca=pca.fit_transform(X_scaled)

print("Explained variance ratio: ", np.round(pca.explained_variance_ratio_, 2))
print("Totalvariance explained: ", round(sum(pca.explained_variance_ratio_)*100,1), "%")
print("Before PCA: ", X_scaled.shape)
print("After: ", X_pca.shape)