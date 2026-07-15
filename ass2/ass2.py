import pandas as pd
import matplotlib.pyplot as plt
import seaborn
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