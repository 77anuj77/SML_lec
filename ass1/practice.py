import pandas as pd
import numpy as np

# 1. Load the Pima Diabetes dataset
df = pd.read_csv('diabetes.csv')

# 2. Display the first 10 records
print("First 10 records:")
print(df.head(10))
print("\n" + "="*50 + "\n")

# 3. Print the shape, column names, and data types
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes)
print("\n" + "="*50 + "\n")

# 4. Generate summary statistics using .describe()
print("Summary Statistics:")
print(df.describe())
print("\n" + "="*50 + "\n")

# 5. Identify columns with physiologically impossible zeros
# Columns where zero values are physiologically impossible: Glucose, BloodPressure, SkinThickness, Insulin, BMI
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("Count of zero values in each column (where zero is physiologically impossible):")
for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    print(f"{col}: {zero_count} zeros")
print("\n" + "="*50 + "\n")

# 6. Replace zeros with NaN in the specified columns
df_clean = df.copy()
for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# 7. Count missing values and compute percentage missing
print("Missing values after replacing zeros with NaN:")
missing_counts = df_clean.isnull().sum()
print(missing_counts)
print("\nPercentage missing:")
missing_percentage = (df_clean.isnull().sum() / len(df_clean)) * 100
print(missing_percentage)
print("\n" + "="*50 + "\n")

# 8. Handle missing values with specific strategies
# Glucose, BloodPressure, BMI → median imputation
for col in ['Glucose', 'BloodPressure', 'BMI']:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# SkinThickness → mean imputation
skin_mean = df_clean['SkinThickness'].mean()
df_clean['SkinThickness'] = df_clean['SkinThickness'].fillna(skin_mean)

# Insulin → median computed per Outcome group (groupwise imputation)
df_clean['Insulin'] = df_clean.groupby('Outcome')['Insulin'].transform(
    lambda x: x.fillna(x.median())
)

# 9. Confirm zero missing values remain after imputation
print("Missing values after imputation:")
print(df_clean.isnull().sum().sum())
print("All missing values handled:", df_clean.isnull().sum().sum() == 0)
print("\n" + "="*50 + "\n")

# 10. Filter and display records where Glucose > 140 AND BMI > 30 AND Outcome == 1
filtered_df = df_clean[(df_clean['Glucose'] > 140) & (df_clean['BMI'] > 30) & (df_clean['Outcome'] == 1)]
print("Records where Glucose > 140 AND BMI > 30 AND Outcome == 1:")
print(filtered_df)
print(f"\nCount of matching records: {len(filtered_df)}")
print("\n" + "="*50 + "\n")

# 11. Using groupby on Outcome column, compute mean of Glucose, BMI, and Age for each group
grouped_means = df_clean.groupby('Outcome')[['Glucose', 'BMI', 'Age']].mean()
print("Mean Glucose, BMI, and Age for each Outcome group:")
print(grouped_means)
print("\n" + "="*50 + "\n")

# Conceptual questions answers (to be written in notebook)
print("CONCEPTUAL QUESTIONS ANSWERS:")
print("1. Why is group-wise median imputation for Insulin better than using a single global median across all patients?")
print("   Answer: Insulin levels likely differ significantly between diabetic and non-diabetic patients. Using group-wise median")
print("   preserves these differences, preventing bias that would occur if we used a single global median that doesn't account for")
print("   the underlying physiological differences between the two groups.")
print("\n2. What does the groupby result in the last step tell you about the difference between diabetic and non-diabetic patients?")
print("   Answer: The groupby results show that diabetic patients (Outcome=1) have higher mean Glucose levels compared to")
print("   non-diabetic patients (Outcome=0), indicating that Glucose is a strong predictor for diabetes diagnosis.")