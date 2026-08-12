"""
Today we will apply Naive Bayes three times:

1. Removing categorical data
2. Including categorical data
3. Changing var_smoothing
"""

import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "/Users/ayushparoha/Downloads/sml/ass3/medicinal drug dataset.csv"
)

print(df.head())

print("\nDataset Information:")
df.info()


# ============================================================
# TARGET VARIABLE: DRUG
# ============================================================

print("\nDrug class distribution:")
print(df["Drug"].value_counts())

print("\nPercentage:")
print(df["Drug"].value_counts(normalize=True) * 100)


# Encode target variable
le_drug = LabelEncoder()

y = le_drug.fit_transform(df["Drug"])

print("\nDrug classes:")
print(le_drug.classes_)


# ============================================================
# ATTEMPT 1
# NAIVE BAYES WITHOUT CATEGORICAL DATA
# ============================================================

print("\n" + "=" * 60)
print("ATTEMPT 1 - WITHOUT CATEGORICAL DATA")
print("=" * 60)


# Only numerical features
X1 = df[["Age", "Na_to_K"]].values

print("\nFeature matrix shape:")
print(X1.shape)


# Train-test split
X1_train, X1_test, y_train, y_test = train_test_split(
    X1,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Standardization
sc1 = StandardScaler()

X1_train = sc1.fit_transform(X1_train)
X1_test = sc1.transform(X1_test)


# Gaussian Naive Bayes
clf1 = GaussianNB()

clf1.fit(X1_train, y_train)


# Prediction
y_pred1 = clf1.predict(X1_test)


# Accuracy
acc1 = accuracy_score(y_test, y_pred1)

print(f"\nAttempt 1 Accuracy: {acc1 * 100:.2f}%")


# Classification report
print("\nClassification Report - Attempt 1:")

print(
    classification_report(
        y_test,
        y_pred1,
        target_names=le_drug.classes_
    )
)


# ============================================================
# ATTEMPT 2
# NAIVE BAYES INCLUDING CATEGORICAL DATA
# ============================================================

print("\n" + "=" * 60)
print("ATTEMPT 2 - INCLUDING CATEGORICAL DATA")
print("=" * 60)


# Make a copy
df_enc = df.copy()


# ------------------------------------------------------------
# Encode categorical columns
# ------------------------------------------------------------

# Find categorical columns
categorical_columns = df_enc.select_dtypes(
    include=["object", "category"]
).columns

print("\nCategorical columns:")
print(categorical_columns)


# Encode each categorical column separately
for col in categorical_columns:

    # Don't encode Drug because it is our target
    if col != "Drug":

        le = LabelEncoder()

        df_enc[col] = le.fit_transform(df_enc[col])


print("\nEncoded dataset:")
print(df_enc.head())


# ------------------------------------------------------------
# Create X2 and y
# ------------------------------------------------------------

# Remove target column
X2 = df_enc.drop(columns="Drug")

# Convert DataFrame to NumPy AFTER encoding
X2 = X2.values

print("\nX2 shape:")
print(X2.shape)


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X2_train, X2_test, y_train, y_test = train_test_split(
    X2,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# Standardization
# ------------------------------------------------------------

sc2 = StandardScaler()

X2_train = sc2.fit_transform(X2_train)
X2_test = sc2.transform(X2_test)


# ------------------------------------------------------------
# Gaussian Naive Bayes
# ------------------------------------------------------------

clf2 = GaussianNB()

clf2.fit(X2_train, y_train)


# Prediction
y_pred2 = clf2.predict(X2_test)


# Accuracy
acc2 = accuracy_score(y_test, y_pred2)

print(f"\nAttempt 2 Accuracy: {acc2 * 100:.2f}%")


# Classification report
print("\nClassification Report - Attempt 2:")

print(
    classification_report(
        y_test,
        y_pred2,
        target_names=le_drug.classes_
    )
)


# ============================================================
# ATTEMPT 3
# DIFFERENT VAR_SMOOTHING VALUES
# ============================================================

print("\n" + "=" * 60)
print("ATTEMPT 3 - VARIANCE SMOOTHING")
print("=" * 60)


# Different smoothing values
smoothing_values = [
    1e-9,
    1e-7,
    1e-5,
    1e-3,
    0.01,
    0.1,
    1.0
]


print(f"{'var_smoothing':<20} {'Accuracy':>10}")
print("-" * 32)


best_accuracy = 0
best_vs = None


for vs in smoothing_values:

    # Create model
    clf_vs = GaussianNB(
        var_smoothing=vs
    )

    # Train
    clf_vs.fit(
        X2_train,
        y_train
    )

    # Predict
    y_pred_vs = clf_vs.predict(
        X2_test
    )

    # Accuracy
    acc_vs = accuracy_score(
        y_test,
        y_pred_vs
    )

    print(
        f"{vs:<20} {acc_vs * 100:>9.2f}%"
    )


    # Find best value
    if acc_vs > best_accuracy:

        best_accuracy = acc_vs
        best_vs = vs


# ============================================================
# BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)


print(f"\nBest var_smoothing: {best_vs}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")


# Train final model using best smoothing
clf3 = GaussianNB(
    var_smoothing=best_vs
)

clf3.fit(
    X2_train,
    y_train
)


# Prediction
y_pred3 = clf3.predict(
    X2_test
)


# Accuracy
acc3 = accuracy_score(
    y_test,
    y_pred3
)

print(
    f"\nAttempt 3 Accuracy: {acc3 * 100:.2f}%"
)


# Classification report
print("\nClassification Report - Attempt 3:")

print(
    classification_report(
        y_test,
        y_pred3,
        target_names=le_drug.classes_
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)


# Attempt 1
cm1 = confusion_matrix(
    y_test,
    y_pred1
)

print("\nConfusion Matrix - Attempt 1:")
print(cm1)


# Attempt 2
cm2 = confusion_matrix(
    y_test,
    y_pred2
)

print("\nConfusion Matrix - Attempt 2:")
print(cm2)


# Attempt 3
cm3 = confusion_matrix(
    y_test,
    y_pred3
)

print("\nConfusion Matrix - Attempt 3:")
print(cm3)

