import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load dataset.csv
df = pd.read_csv('dataset.csv')

# Display first 10 rows
print("First 10 rows:")
print(df.head(10))
print("\n" + "="*50 + "\n")

# Display shape
print("Shape of dataset:", df.shape)
print("\n" + "="*50 + "\n")

# Display data types
print("Data types:")
print(df.dtypes)
print("\n" + "="*50 + "\n")

# Display summary statistics
print("Summary statistics:")
print(df.describe())
print("\n" + "="*50 + "\n")

# Drop unnecessary columns
cols_to_drop = ['track_id', 'artists', 'album_name', 'track_name']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

# Check for missing values
print("Missing values before handling:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")

# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())
print("\n" + "="*50 + "\n")

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values: drop rows with any missing values
df = df.dropna()

print("Shape after dropping duplicates and missing values:", df.shape)
print("\n" + "="*50 + "\n")

# Plot histograms for danceability, energy, and tempo
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['danceability'], kde=True)
plt.title('Danceability Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['energy'], kde=True)
plt.title('Energy Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['tempo'], kde=True)
plt.title('Tempo Distribution')
plt.tight_layout()
plt.show()

# Create popularity column
df['popularity'] = pd.cut(df['popularity'], bins=[-np.inf, 40, 70, np.inf], labels=['low', 'medium', 'high'])

# Plot bar chart showing mean energy per popularity group
plt.figure(figsize=(8, 6))
sns.barplot(x='popularity', y='energy', data=df, ci=None)
plt.title('Mean Energy per Popularity Level')
plt.ylabel('Mean Energy')
plt.show()

# Plot scatter plot of danceability vs energy, coloured by popularity
plt.figure(figsize=(8, 6))
sns.scatterplot(x='danceability', y='energy', hue='popularity', data=df, palette='viridis')
plt.title('Danceability vs Energy by Popularity Level')
plt.show()

# Plot box plot comparing loudness across popularity groups
plt.figure(figsize=(8, 6))
sns.boxplot(x='popularity', y='loudness', data=df)
plt.title('Loudness Distribution by Popularity Level')
plt.show()

# Plot correlation heatmap (lower triangle only, annotated) for all numeric features
# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Heatmap (Lower Triangle)')
plt.show()

# Identify the strongest correlated pair (excluding self-correlation)
# We'll find the maximum absolute correlation in the upper triangle (lower triangle? Actually we masked upper, so we look at lower triangle without diagonal)
# Compute the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Take absolute values
abs_corr = corr_matrix.abs()

# Ignore self-correlation
np.fill_diagonal(abs_corr.values, 0)

# Get the maximum absolute correlation
max_corr = abs_corr.max().max()
# Get the indices of the maximum correlation
i, j = np.where(abs_corr == max_corr)
# If multiple, take the first pair
if len(i) > 0:
    row, col = i[0], j[0]
    var1, var2 = corr_matrix.columns[row], corr_matrix.columns[col]
    corr_value = corr_matrix.iloc[row, col]
    print(f"Strongest correlation: {var1} and {var2} = {corr_value:.4f}")
else:
    print("No correlation found.")
print("\n" + "="*50 + "\n")

# Apply Label Encoding to explicit column (True/False)
if 'explicit' in df.columns:
    le = LabelEncoder()
    df['explicit_encoded'] = le.fit_transform(df['explicit'].astype(str))
else:
    print("Column 'explicit' not found.")
    # Create a dummy column if not present
    df['explicit_encoded'] = 0

# Apply One-Hot Encoding with drop_first=True
if 'popularity' in df.columns:
    ohe = OneHotEncoder(drop='first', sparse_output=False)

    popularity_encoded = ohe.fit_transform(df[['popularity']])

    popularity_encoded_df = pd.DataFrame(
        popularity_encoded,
        columns=ohe.get_feature_names_out(['popularity']),
        index=df.index
    )

    df = pd.concat([df, popularity_encoded_df], axis=1)
else:
    print("Column 'popularity' not found.")

# Create two new ratio features
df['energy_dance_ratio'] = df['energy'] / (df['danceability'] + 1e-6)
df['acousticness_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)

# Bin tempo into 3 equal-width categories using pd.cut() with labels slow, medium, fast
df['tempo_bin'] = pd.cut(df['tempo'], bins=3, labels=['slow', 'medium', 'fast'])
print("Count per tempo bin:")
print(df['tempo_bin'].value_counts().sort_index())
print("\n" + "="*50 + "\n")

# Compute the absolute correlation matrix
# We'll consider only numeric columns for correlation (including the new encoded and ratio features)
# But note: we have added categorical encoded columns (explicit_encoded, popularity_encoded_*) which are numeric (0/1)
# We'll select all numeric columns again
numeric_cols_after = df.select_dtypes(include=[np.number]).columns
corr_matrix_abs = df[numeric_cols_after].corr().abs()

# We want to find pairs with correlation > 0.85 (excluding self-correlation and duplicates)
# We'll create a mask for the upper triangle without diagonal
upper_tri = np.triu(np.ones_like(corr_matrix_abs, dtype=bool), k=1)
high_corr_pairs = []
for i in range(len(corr_matrix_abs.columns)):
    for j in range(i+1, len(corr_matrix_abs.columns)):
        if corr_matrix_abs.iloc[i, j] > 0.85:
            high_corr_pairs.append((corr_matrix_abs.columns[i], corr_matrix_abs.columns[j], corr_matrix_abs.iloc[i, j]))

if high_corr_pairs:
    print("Features with correlation > 0.85:")
    for var1, var2, corr in high_corr_pairs:
        print(f"{var1} & {var2}: {corr:.4f}")
else:
    print("No feature pairs with correlation > 0.85.")
print("\n" + "="*50 + "\n")

# Standardise all numeric features using StandardScaler
scaler = StandardScaler()
numeric_data = df[numeric_cols_after]
scaled_data = scaler.fit_transform(numeric_data)
scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols_after)

print("Dataset dimensions before scaling:", numeric_data.shape)
print("Dataset dimensions after scaling:", scaled_df.shape)
print("\n" + "="*50 + "\n")

# Apply PCA with 5 components
pca = PCA(n_components=5)

# Fit PCA and transform the data
pca_result = pca.fit_transform(scaled_df)

# Convert to DataFrame
pca_df = pd.DataFrame(
    pca_result,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print("Dataset dimensions before PCA:", scaled_df.shape)
print("Dataset dimensions after PCA:", pca_df.shape)