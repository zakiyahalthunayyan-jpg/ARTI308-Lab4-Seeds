# ===============================
# 1) Import Libraries
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ===============================
# 2) Load Dataset
# ===============================
df = pd.read_csv("seeds_dataset.csv", sep=r"\s+", header=None, engine="python")
df.columns = ["Area", "Perimeter", "Compactness", "Length", "Width", "Asymmetry", "Groove", "Class"]


# ===============================
# Task 2) Create Missing Values (for demonstration) + Handle them
# ===============================
# ملاحظة: Seeds dataset غالبًا ما فيها قيم مفقودة، لذلك نضيف Missing Values للتجربة فقط
df.loc[np.random.choice(df.index, 5, replace=False), "Area"] = np.nan
print("Note: Missing values were introduced intentionally for demonstration.\n")

print("Missing Values After Creation:")
print(df.isnull().sum())

# استراتيجية معالجة القيم المفقودة: Mean Imputation (لأن العمود عددي)
df["Area"] = df["Area"].fillna(df["Area"].mean())
print("\nMissing Values After Imputation:")
print(df.isnull().sum())


# ===============================
# Task 1) Data Exploration + Quality Checks
# ===============================
print("\nFirst 5 rows:\n")
print(df.head())

print("\nDataset Shape (rows, columns):")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values (Final Check):")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ===============================
# Split Features and Target
# ===============================
X = df.drop("Class", axis=1)
y = df["Class"]


# ===============================
# Task 3) Outliers Detection + Handling using IQR
# ===============================
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1

outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))

print("\nNumber of Outliers per Feature:")
print(outliers.sum())

# إزالة الصفوف التي تحتوي على outliers في أي عمود
mask_no_outliers = ~outliers.any(axis=1)
X_no_outliers = X[mask_no_outliers]
y_no_outliers = y[mask_no_outliers]

print("\nShape After Removing Outliers (X, y):")
print(X_no_outliers.shape, y_no_outliers.shape)


# ===============================
# Boxplot (Before/After Outlier Removal)
# ===============================
plt.figure(figsize=(10, 6))
sns.boxplot(data=X)
plt.xticks(rotation=90)
plt.title("Boxplot (Before Outlier Removal)")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=X_no_outliers)
plt.xticks(rotation=90)
plt.title("Boxplot (After Outlier Removal)")
plt.show()


# ===============================
# Task 4) Z-Score Standardization (StandardScaler)
# ===============================
std_scaler = StandardScaler()
X_standardized = std_scaler.fit_transform(X_no_outliers)

print("\nStandardization (Z-score) completed.")
print("Mean after standardization:", X_standardized.mean(axis=0))
print("Std after standardization:", X_standardized.std(axis=0))


# ===============================
# Task 4) Min-Max Normalization (MinMaxScaler)
# ===============================
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_no_outliers)

print("\nMin-Max Normalization completed.")
print("Minimum values after normalization:", X_minmax.min(axis=0))
print("Maximum values after normalization:", X_minmax.max(axis=0))


# ===============================
# Task 5) Correlation Heatmap (Check correlation before PCA)
# ===============================
plt.figure(figsize=(8, 6))
sns.heatmap(X_no_outliers.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# نحدد إذا فيه ارتباط قوي بين الميزات قبل تطبيق PCA
corr_matrix = X_no_outliers.corr().abs()
corr_vals = corr_matrix.to_numpy().copy()
np.fill_diagonal(corr_vals, 0)
max_corr = corr_vals.max()
max_corr = corr_matrix.max().max()

print("\nMax absolute correlation between features:", max_corr)

principal_components = None
pca = None

if max_corr > 0.8:
    print("High correlation detected -> Applying PCA...")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_standardized)
    print("Explained Variance Ratio (PCA):", pca.explained_variance_ratio_)
else:
    print("No strong correlation detected -> Skipping PCA.")


# ===============================
# PCA Scatter Plot (Only if PCA was applied)
# ===============================
if principal_components is not None:
    plt.figure(figsize=(6, 4))
    plt.scatter(principal_components[:, 0],
                principal_components[:, 1],
                c=y_no_outliers,
                cmap="viridis")

    plt.title("PCA Projection (2 Components)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label="Class")
    plt.show()