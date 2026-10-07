import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "/Users/ayushparoha/Downloads/sml/ass6/ipl_matches (1).csv"
)

print("Shape:", df.shape)
print("Columns:", list(df.columns))


# ============================================================
# 2. BASIC DATA ANALYSIS
# ============================================================

cat_cols = df.select_dtypes(include=["object"])

print("\nMatch outcomes:")
print(df["team_win"].value_counts())

print(
    f"\nTeam 1 wins: "
    f"{df['team_win'].mean():.1%} of the time"
)

print(f"\nTeams: {df['team1'].nunique()}")
print(f"Venues: {df['venue'].nunique()}")

print(
    f"\nSeasons: "
    f"{[int(s) for s in sorted(df['seasons'].unique())]}"
)


# ============================================================
# 3. ENCODE CATEGORICAL COLUMNS
# ============================================================

df_enc = df.copy()

categorical_columns = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "venue"
]

for col in categorical_columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])


# ============================================================
# 4. CREATE FEATURES AND TARGET
# ============================================================

X = df_enc.drop(columns="team_win")
y = df_enc["team_win"]

print("\nFeature matrix shape:", X.shape)
print("Features:", list(X.columns))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain:", X_train.shape)
print("Test:", X_test.shape)


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)


# ============================================================
# 7. PREDICTION AND ACCURACY
# ============================================================

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {acc * 100:.2f}%")
print(f"OOB Score: {rf.oob_score_ * 100:.2f}%")


# ============================================================
# 8. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Team 1 Losses",
            "Team 1 Wins"
        ]
    )
)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=[
        "Team 1 Losses",
        "Team 1 Wins"
    ],
    cmap="Blues"
)

plt.title("Random Forest - IPL")
plt.tight_layout()
plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(8, 5))

importance.sort_values().plot(
    kind="barh",
    edgecolor="white"
)

plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("What Decides an IPL Match?")
plt.tight_layout()
plt.show()


print("\nTop 3 most influential factors:")

print(
    importance
    .sort_values(ascending=False)
    .head(3)
)


# ============================================================
# 11. ROC CURVE AND ROC-AUC
# ============================================================

y_prob = rf.predict_proba(X_test)[:, 1]

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    plot_chance_level=True
)

plt.title("ROC Curve - Random Forest")
plt.tight_layout()
plt.show()


print(
    f"\nAccuracy: "
    f"{accuracy_score(y_test, rf.predict(X_test)):.2f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc_score(y_test, y_prob):.4f}"
)


# ============================================================
# 12. BIAS-VARIANCE / NUMBER OF TREES
# ============================================================

n_trees = [1, 5, 10, 20, 50, 100, 150, 200]

classifiers = []

for n in n_trees:

    clf = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    clf.fit(X_train, y_train)

    classifiers.append(clf)


train_accuracies = [
    clf.score(X_train, y_train) * 100
    for clf in classifiers
]

test_accuracies = [
    clf.score(X_test, y_test) * 100
    for clf in classifiers
]


plt.figure(figsize=(8, 5))

plt.plot(
    n_trees,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    n_trees,
    test_accuracies,
    marker="s",
    label="Test Accuracy"
)

plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Accuracy (%)")
plt.title("Bias-Variance Trade-off")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#ensemble algorithm : xgboost, Random forest and these algorithm use boot straping the rason for the boot straping is to avoid the mistake o desion treewe also replace rows and cols
#in RF ooB: Out of Bag score and accuracy(accuracy of Each Decision Tree)
'''🎯 What is OOB Score?
In Random Forest, each tree is built on a random subset of the data (called the “out-of-bag” sample — the data not used to train that tree). Since each tree is trained on a bootstrap sample (with replacement), the remaining ~36.8% of data (the OOB sample) is not used during training for that tree.

📌 OOB = Out-of-Bag — data not used to train a particular tree.

The OOB score is computed by predicting the OOB samples using all trees (i.e., the ensemble) — and then measuring performance (typically using mean squared error for regression or accuracy for classification).

📊 Why is it Useful?
No need for explicit cross-validation (like K-Fold CV)
Faster to compute — since OOB data is already computed during training
Provides an unbiased estimate of performance'''

rf= RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred= rf.predict(X_test)
acc= accuracy_score(y_test, y_pred)

print(f"Test accuracy: , {acc*100:.2f}%")
print(f"OOB score: , {rf.oob_score_*100:.2f}%")

print(classification_report(y_test, y_pred, target_names=['Team 1 Losses', 'Team 1 Wins']))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Team 1 Losses', 'Team 1 Wins'],
    cmap='Blues'
)

plt.tittle("Random Forest - IPL")
plt.tight_layout()
plt.show()

importance= pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8,5))
importance.plot(kind='barh', color='steelblue', edgecolor='white')
plt.xlabel("Feature inportance score")
plt.ylabel("what decides an IPL Match")
plt.tight_layout()
plt.show()

print("Top 3 most influential factors: ")
print(importance.sort_values(ascending=False.head(3)))


from sklearn.metrics import RocCurveDisplay, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Predict probabilities for positive class
y_pred = rf.predict_proba(X_test)[:, 1]

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, y_pred, color="darkorange", plot_chance_level=True)

# Print Accuracy
print(f"Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2f}")

# Print ROC-AUC
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")


n_trees=[1,5,10,20,50,100,150,200]
clfs=[RandomForestClassifier(n_estimators=n, random_state=42,).fit(X_train, y_train) for n in n_trees]
train_acces= [clf.score(X_train, y_train)*100 for clf in clfs]
test_acces= [clf.score(X_test, y_test)*100 for clf in clfs]

plt.plot(n_trees, train_acces, marker='a', label="train Accuracy")
plt.plot(n_trees, test_acces, marker='s', label= "Test Accuraxy")
plt.xlabel("Numer of trees (n_estimators)")
plt.ylabel("Accuracy (%)")
plt.title("Bias barience Trade-off")
plt.legend()
plt.show()